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
            # Be defensive about missing keys
            kind = b.get("kind") or "text"   # default to "text" if None
            text_main = (b.get("text_main") or "").strip()
            narrative = (b.get("narrative") or "").strip()

            # ---- What we actually embed ----
            # For figures/tables, prefer the Gemini narrative if present.
            # For normal text, fall back to text_main.
            if kind in ("image", "figure", "table") and narrative:
                content_for_index = narrative
            else:
                content_for_index = text_main

            if not content_for_index:
                # Nothing meaningful to index; skip this bundle
                continue

            # Build safe metadata: no None values
            meta = {
                "doc_id": doc_id,
                "children": b.get("children", []),
            }

            cost = b.get("cost")
            if cost is not None:
                meta["tokens_estimate"] = cost

            if kind:
                meta["kind"] = kind

            # Keep raw fields only if non-empty strings
            if text_main:
                meta["text_main"] = text_main
            if narrative:
                meta["narrative"] = narrative

            vecs.append(
                {
                    "id": b["id"],
                    # This "text" is what pinecone_index.upsert() will embed.
                    "text": content_for_index,
                    "metadata": meta,
                }
            )

        if not vecs:
            print(f"   WARNING: No vectors built for {doc_id}")
            continue

        pine_upsert(doc_id, vecs)
        print(f"   Bundles: {len(bundles)}, Indexed: {len(vecs)}")


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
        print('  python cli.py ask "your question" [--budget 3000]')
        sys.exit(1)

    cmd = sys.argv[1]

    if cmd == "ingest":
        folder = sys.argv[2] if len(sys.argv) > 2 else "data/papers"
        ingest(folder)

    elif cmd == "ask":
        budget = CONF.token_budget
        if "--budget" in sys.argv:
            try:
                budget = int(sys.argv[sys.argv.index("--budget") + 1])
            except Exception:
                pass
        q_parts = [x for x in sys.argv[2:] if x not in ("--budget", str(budget))]
        ask(" ".join(q_parts), budget)

    else:
        print("Unknown command")


if __name__ == "__main__":
    main()
