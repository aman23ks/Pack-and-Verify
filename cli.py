#!/usr/bin/env python3
import sys, os, pathlib
from pav.config import CONF
from pav.clients.unstructured_api import partition_pdf
from pav.clients.context_enrichment import enrich_with_context
from pav.ingest.build_bundles import build as build_bundles
from pav.clients.pinecone_index import upsert as pine_upsert
from pav.clients.retrieval.search import search
from pav.clients.retrieval.packer import pack, render_pack
from pav.qa.answer import answer

def ingest(folder: str):
    src = pathlib.Path(folder)
    assert src.exists(), f"Folder not found: {folder}"
    for pdf in sorted(src.glob("*.pdf")):
        doc_id = pdf.stem
        print(f"→ Ingest {pdf.name}")
        elems = partition_pdf(str(pdf))
        elems = enrich_with_context(elems, neighbors=3)
        bundles = build_bundles(elems, doc_id)
        vecs = []
        for b in bundles:
            text = "\n".join([*b["context_pre"], b["text_main"], b["context_post"]]).strip()
            vecs.append({"id": b["id"], "text": text, "metadata": {
                "doc_id": doc_id, "page": b["page"], "kind": b["kind"],
                "text_main": b["text_main"], "context_pre": b["context_pre"],
                "context_post": b["context_post"], "tokens_estimate": b["cost"]
            }})
        pine_upsert(doc_id, vecs)
        print(f"   Bundles: {len(bundles)}")

def ask(question: str, budget: int, doc_filter: str | None = None):
    """
    Ask a question against the vector store.
    If doc_filter is provided, restrict results to that doc_id/namespace.
    """
    # Prefer the retrieval layer to do namespace filtering if it supports it.
    try:
        matches = search(question, top_k=80, doc_id=doc_filter)  # if your search() supports doc_id
    except TypeError:
        matches = search(question, top_k=80)  # fallback; we'll filter locally below

    # Local filter by metadata.doc_id if needed
    if doc_filter:
        filtered = []
        for m in matches:
            md = m.get("metadata") if isinstance(m, dict) else getattr(m, "metadata", {}) or {}
            if (md.get("doc_id") or "").strip() == doc_filter:
                filtered.append(m)
        if filtered:
            matches = filtered
            
    chosen, used = pack(matches, budget)
    pack_text = render_pack(chosen)
    ans = answer(pack_text, question)
    print(f"\n[PACK TOKENS ≈ {used}]")
    print("\n=== ANSWER ===\n" + ans)

def _print_usage():
    print("Usage:")
    print("  python cli.py ingest data/papers")
    print("  python cli.py ask \"your question\" [--budget 3000] [--doc DOC_ID]")

def main():
    if len(sys.argv) < 2:
        _print_usage()
        sys.exit(1)

    cmd = sys.argv[1]
    if cmd == "ingest":
        folder = sys.argv[2] if len(sys.argv) > 2 else "data/papers"
        ingest(folder)
    elif cmd == "ask":
        budget = CONF.token_budget
        doc_filter = None

        if "--budget" in sys.argv:
            try:
                budget = int(sys.argv[sys.argv.index("--budget") + 1])
            except Exception:
                pass

        if "--doc" in sys.argv:
            try:
                doc_filter = sys.argv[sys.argv.index("--doc") + 1]
            except Exception:
                pass

        # Remove flags from the question assembly
        skip = set()
        if "--budget" in sys.argv:
            skip.update(["--budget", str(budget)])
        if "--doc" in sys.argv and doc_filter:
            skip.update(["--doc", doc_filter])

        q_parts = [x for x in sys.argv[2:] if x not in skip]
        question = " ".join(q_parts).strip()
        if not question:
            _print_usage()
            sys.exit(1)

        ask(question, budget, doc_filter)
    else:
        print("Unknown command")
        _print_usage()

if __name__ == "__main__":
    main()
