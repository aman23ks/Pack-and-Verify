#!/usr/bin/env python3
import sys
import pathlib

from pav.config import CONF
from pav.clients.unstructured_api import partition_pdf
from pav.ingest.build_bundles import build as build_bundles
from pav.clients.pinecone_index import upsert as pine_upsert
from pav.clients.retrieval.search import search
from pav.clients.retrieval.packer import pack, render_pack
from pav.qa.answer import answer
import json
from pav.clients.rerank_local import local_rerank, RERANK_AVAILABLE
import os
import time



def ingest(folder: str):
    src = pathlib.Path(folder)
    assert src.exists(), f"Folder not found: {folder}"

    for pdf in sorted(src.glob("*.pdf")):
        doc_id = pdf.stem
        print(f"→ Ingest {pdf.name}")

        # 1) Run unstructured on the PDF
        elems = partition_pdf(str(pdf))
        
        # 2) Build logical bundles (sections, figures, tables, etc.)
        bundles = build_bundles(elems, doc_id)
        
        vecs = []
        for b in bundles:
            # Original metadata from discourse_units
            src_meta = b.get("metadata") or {}

            # Prefer top-level kind if present, else metadata.kind, else "text"
            kind = (b.get("kind") or src_meta.get("kind") or "text")
            kind_lower = (kind or "").lower()

            # Text fields: prefer top-level; fall back to metadata if needed
            text_main = (b.get("text_main") or src_meta.get("text_main") or "").strip()
            narrative = (b.get("narrative") or src_meta.get("narrative") or "").strip()

            # ---- What we actually embed ----
            # For figures/tables, prefer the narrative if present.
            if kind_lower in ("image", "figure", "table") and narrative:
                content_for_index = narrative
            else:
                content_for_index = text_main

            if not content_for_index:
                # Nothing meaningful to index; skip this bundle
                continue

            # Children may live either at top-level or inside metadata
            children = b.get("children")
            if children is None:
                children = src_meta.get("children", [])

            # Base metadata (never None values)
            meta = {
                "doc_id": doc_id,
                "children": children,
            }

            # Cost / token estimate
            cost = b.get("cost")
            if cost is None:
                cost = src_meta.get("tokens_estimate")
            if cost is not None:
                meta["tokens_estimate"] = cost

            # Page (this is what packer needs)
            page = src_meta.get("page")
            if page is not None:
                meta["page"] = page

            # Kind
            if kind_lower:
                meta["kind"] = kind_lower

            # Keep raw fields only if non-empty strings
            if text_main:
                meta["text_main"] = text_main
            if narrative:
                meta["narrative"] = narrative

            # Optional extra fields
            for extra_key in ("element_id", "parent_ccu", "caption"):
                v = src_meta.get(extra_key)
                if v is not None:
                    meta[extra_key] = v

            vecs.append(
                {
                    "id": b["id"],
                    "text": content_for_index,  # what pinecone_index.upsert() embeds
                    "metadata": meta,
                }
            )

        if not vecs:
            print(f"   WARNING: No vectors built for {doc_id}")
            continue

        pine_upsert(doc_id, vecs)
        print(f"   Bundles: {len(bundles)}, Indexed: {len(vecs)}")


def ask(question: str, budget: int, namespace: str | None = None):
    # Env toggles:
    # USE_RERANK=0 → disable local cross-encoder rerank
    # USE_PENALTY=0 → disable cosine diversity penalty in pack()
    # PROFILE=1     → print timing info per call
    use_rerank = os.getenv("USE_RERANK", "1") != "0"
    use_penalty = os.getenv("USE_PENALTY", "1") != "0"
    do_profile = os.getenv("PROFILE", "0") == "1"

    t0 = time.perf_counter()

    # 1) Retrieve from Pinecone
    matches = search(question, top_k=80, namespace=namespace)

    # 2) Optional: local rerank
    if use_rerank and RERANK_AVAILABLE:
        matches = local_rerank(question, matches)

    # 3) Pack with or without redundancy penalty
    lambda_penalty = 0.3 if use_penalty else 0.0
    chosen, used = pack(matches, budget, lambda_penalty=lambda_penalty)

    # 4) Build context and answer
    pack_text = render_pack(chosen)
    ans = answer(pack_text, question)

    if do_profile:
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        print(
            f"[PROFILE] use_rerank={use_rerank} "
            f"use_penalty={use_penalty} "
            f"budget={budget} tokens≈{used} time_ms={elapsed_ms:.1f}"
        )

    # Keep the prints for interactive use
    print(f"\n[PACK TOKENS ≈ {used}]")
    print("\n=== ANSWER ===\n" + ans)

    # IMPORTANT: return the answer so batch_eval can capture it
    return ans

def main():
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python cli.py ingest data/papers")
        print('  python cli.py ask "your question" [--budget 3000] [--ns DOC_ID]')
        sys.exit(1)

    cmd = sys.argv[1]

    if cmd == "ingest":
        folder = sys.argv[2] if len(sys.argv) > 2 else "data/papers"
        ingest(folder)

    elif cmd == "ask":
        budget = CONF.token_budget
        namespace = None

        # Parse flags manually so we can keep question words free-form
        args = sys.argv[2:]
        i = 0
        question_parts: list[str] = []

        while i < len(args):
            arg = args[i]
            if arg == "--budget" and i + 1 < len(args):
                try:
                    budget = int(args[i + 1])
                except Exception:
                    pass
                i += 2
            elif arg in ("--ns", "--namespace") and i + 1 < len(args):
                namespace = args[i + 1]
                i += 2
            else:
                question_parts.append(arg)
                i += 1

        question = " ".join(question_parts).strip()
        if not question:
            print("No question provided.")
            sys.exit(1)

        # When called from CLI, we ignore the returned value and just print
        ask(question, budget, namespace)

    else:
        print("Unknown command")


if __name__ == "__main__":
    main()
