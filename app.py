"""
Pack-and-Verify – FastAPI backend
Wraps the existing CLI functionality (ingest + ask) as REST endpoints.
"""
import os
import pathlib
import shutil
import threading
import tempfile
from typing import Optional, List

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from pav.config import CONF
from pav.clients.unstructured_api import partition_pdf
from pav.ingest.build_bundles import build as build_bundles
from pav.clients.pinecone_index import upsert as pine_upsert
from pav.clients.retrieval.search import search
from pav.clients.retrieval.packer import pack, render_pack
from pav.qa.answer import answer as qa_answer
from pav.clients.rerank_local import local_rerank, RERANK_AVAILABLE

# ── App ──────────────────────────────────────────────────────────────
app = FastAPI(title="Pack-and-Verify API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Track ingested documents (namespace = doc_id = PDF stem) ────────
# These are the logical names of documents successfully sent to Pinecone.
# In a production app, you might save these to a small DB/file, 
# but in-memory is perfect for a clean, cloud-native demo.
ingested_namespaces: List[str] = []

# ── Background ingest state ─────────────────────────────────────────
ingest_state = {
    "running": False,
    "status": "idle",
    "detail": "",
}


def _ingest_single_pdf(pdf_path: pathlib.Path):
    """Ingest one PDF: parse → bundle → embed → upsert."""
    doc_id = pdf_path.stem
    print(f"→ Ingest {pdf_path.name}")

    elems = partition_pdf(str(pdf_path))
    bundles = build_bundles(elems, doc_id)

    vecs = []
    for b in bundles:
        src_meta = b.get("metadata") or {}
        kind = (b.get("kind") or src_meta.get("kind") or "text")
        kind_lower = (kind or "").lower()
        text_main = (b.get("text_main") or src_meta.get("text_main") or "").strip()
        narrative = (b.get("narrative") or src_meta.get("narrative") or "").strip()

        if kind_lower in ("image", "figure", "table") and narrative:
            content_for_index = narrative
        else:
            content_for_index = text_main

        if not content_for_index:
            continue

        children = b.get("children")
        if children is None:
            children = src_meta.get("children", [])

        meta = {"doc_id": doc_id, "children": children}

        cost = b.get("cost")
        if cost is None:
            cost = src_meta.get("tokens_estimate")
        if cost is not None:
            meta["tokens_estimate"] = cost

        page = src_meta.get("page")
        if page is not None:
            meta["page"] = page
        if kind_lower:
            meta["kind"] = kind_lower
        if text_main:
            meta["text_main"] = text_main
        if narrative:
            meta["narrative"] = narrative

        for extra_key in ("element_id", "parent_ccu", "caption"):
            v = src_meta.get(extra_key)
            if v is not None:
                meta[extra_key] = v

        vecs.append({"id": b["id"], "text": content_for_index, "metadata": meta})

    if not vecs:
        print(f"   WARNING: No vectors built for {doc_id}")
        return

    pine_upsert(doc_id, vecs)
    print(f"   Bundles: {len(bundles)}, Indexed: {len(vecs)}")

    # Track this namespace
    if doc_id not in ingested_namespaces:
        ingested_namespaces.append(doc_id)


def _run_ingest_background(temp_dir_path: str, filenames: List[str]):
    """Run ingest on specific uploaded files in a background thread."""
    temp_dir = pathlib.Path(temp_dir_path)
    try:
        ingest_state["running"] = True
        ingest_state["status"] = "running"

        for i, fname in enumerate(filenames, 1):
            ingest_state["detail"] = f"Processing {fname} ({i}/{len(filenames)})…"
            pdf_path = temp_dir / fname
            if pdf_path.exists():
                _ingest_single_pdf(pdf_path)

        ingest_state["status"] = "done"
        ingest_state["detail"] = f"Successfully ingested {len(filenames)} file(s)"
    except Exception as e:
        ingest_state["status"] = "error"
        ingest_state["detail"] = str(e)
    finally:
        ingest_state["running"] = False
        # Clean up the hidden temporary workspace Entirely
        shutil.rmtree(temp_dir, ignore_errors=True)


def _ask_with_namespace(question: str, budget: int, namespace: Optional[str] = None) -> str:
    """
    Run the retrieval + QA pipeline.
    If namespace is None, search ALL ingested namespaces and merge results.
    """
    use_rerank = os.getenv("USE_RERANK", "1") != "0"
    use_penalty = os.getenv("USE_PENALTY", "1") != "0"

    # Collect matches across namespace(s)
    all_matches = []

    if namespace:
        # Search specific namespace
        all_matches = search(question, top_k=80, namespace=namespace)
    else:
        # Search ALL ingested namespaces
        namespaces_to_search = ingested_namespaces if ingested_namespaces else [None]
        for ns in namespaces_to_search:
            matches = search(question, top_k=40, namespace=ns)
            all_matches.extend(matches)

    if not all_matches:
        return "No matching documents found. Please upload and ingest documents first."

    # Rerank
    if use_rerank and RERANK_AVAILABLE:
        all_matches = local_rerank(question, all_matches)

    # Pack
    lambda_penalty = 0.3 if use_penalty else 0.0
    chosen, used = pack(all_matches, budget, lambda_penalty=lambda_penalty)

    # Answer
    pack_text = render_pack(chosen)
    ans = qa_answer(pack_text, question)

    print(f"\n[PACK TOKENS ≈ {used}]")
    return ans


# ── Schemas ──────────────────────────────────────────────────────────
class AskRequest(BaseModel):
    question: str
    budget: int = CONF.token_budget
    namespace: Optional[str] = None


class AskResponse(BaseModel):
    answer: str


class StatusResponse(BaseModel):
    status: str
    detail: str = ""


# ── Endpoints ────────────────────────────────────────────────────────
@app.get("/api/health", response_model=StatusResponse)
def health():
    return StatusResponse(status="ok", detail="Pack-and-Verify API is running")


@app.post("/api/upload-and-ingest", response_model=StatusResponse)
async def upload_and_ingest(files: List[UploadFile] = File(...)):
    """Upload PDFs and immediately start ingesting them in the background."""
    if ingest_state["running"]:
        raise HTTPException(status_code=409, detail="Ingestion already in progress. Please wait.")

    # Create a unique temporary directory for this upload batch
    temp_dir = tempfile.mkdtemp()
    temp_path = pathlib.Path(temp_dir)

    saved = []
    for f in files:
        if not f.filename or not f.filename.lower().endswith(".pdf"):
            shutil.rmtree(temp_path, ignore_errors=True)
            raise HTTPException(status_code=400, detail=f"Only PDF files are accepted (got {f.filename})")
        
        dest = temp_path / f.filename
        with dest.open("wb") as out:
            shutil.copyfileobj(f.file, out)
        saved.append(f.filename)

    thread = threading.Thread(target=_run_ingest_background, args=(temp_dir, saved), daemon=True)
    thread.start()

    return StatusResponse(
        status="started",
        detail=f"Uploaded & started ingesting {len(saved)} file(s): {', '.join(saved)}"
    )


@app.get("/api/ingest/status", response_model=StatusResponse)
def ingest_status():
    return StatusResponse(status=ingest_state["status"], detail=ingest_state["detail"])


@app.post("/api/ask", response_model=AskResponse)
def ask_question(req: AskRequest):
    """Run the retrieval + QA pipeline. Searches all namespaces if none specified."""
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")
    try:
        ans = _ask_with_namespace(req.question.strip(), req.budget, req.namespace)
        return AskResponse(answer=ans)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/documents")
def list_documents():
    """List documents already ingested in Pinecone."""
    # Since we delete the physical files, we'll use our in-memory list
    # Let's map the namespace back to a logical filename string
    docs = [f"{ns}.pdf" for ns in ingested_namespaces]
    return {
        "documents": docs,
        "namespaces": list(ingested_namespaces),
    }


@app.get("/")
def root_health_check():
    """Simple root health check so Render's port scanner finds us immediately."""
    return {"status": "alive", "service": "Pack-and-Verify"}


# ── Entrypoint ───────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    # Render provides the port via an environment variable
    port = int(os.environ.get("PORT", 8000))
    # Note: 'reload=True' is great for local dev but should be False on Render
    uvicorn.run("app:app", host="0.0.0.0", port=port, reload=False)
