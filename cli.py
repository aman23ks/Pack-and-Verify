#!/usr/bin/env python3
import sys, os, pathlib
from pav.config import CONF
from pav.clients.unstructured_api import partition_pdf
from pav.ingest.build_bundles import build as build_bundles
from pav.clients.pinecone_index import upsert as pine_upsert
from pav.clients.retrieval.search import search
from pav.clients.retrieval.packer import pack, render_pack
from pav.qa.answer import answer
import json

def ingest(folder: str):
    src = pathlib.Path(folder)
    assert src.exists(), f"Folder not found: {folder}"
    for pdf in sorted(src.glob("*.pdf")):
        doc_id = pdf.stem
        print(f"→ Ingest {pdf.name}")
        elems = partition_pdf(str(pdf))
        bundles = build_bundles(elems, doc_id)

        vecs = []
        for b in bundles:
            # All content is in text_main (section text + [Figures/Tables] block)
            text = b["text_main"].strip()

            vecs.append({
                "id": b["id"],
                "text": text,
                "metadata": {
                    "doc_id": doc_id,
                    "tokens_estimate": b["cost"],
                    "children": b.get("children", []),
                },
            })

        pine_upsert(doc_id, vecs)
        print(f"   Bundles: {len(bundles)}")

def ask(question: str, budget: int):
    matches = search(question, top_k=80)
    chosen, used = pack(matches, budget)
    pack_text = render_pack(chosen)
    ans = answer(pack_text, question)
    print(f"\n[PACK TOKENS ≈ {used}]")
    print("\n=== ANSWER ===\n" + ans)

def main():
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python cli.py ingest data/papers")
        print("  python cli.py ask \"your question\" [--budget 3000]")
        sys.exit(1)
    cmd = sys.argv[1]
    if cmd == "ingest":
        folder = sys.argv[2] if len(sys.argv) > 2 else "data/papers"
        ingest(folder)
    elif cmd == "ask":
        budget = CONF.token_budget
        if "--budget" in sys.argv:
            try: budget = int(sys.argv[sys.argv.index("--budget")+1])
            except: pass
        q_parts = [x for x in sys.argv[2:] if x not in ("--budget", str(budget))]
        ask(" ".join(q_parts), budget)
    else:
        print("Unknown command")

if __name__ == "__main__":
    main()
