# pav/clients/retrieval/packer.py

from typing import List, Tuple, Any


def _extract_text(md: dict) -> str:
    """
    Decide what text to feed into the QA model from a match's metadata.
    """
    return (
        md.get("content")
        or md.get("text_main")
        or md.get("narrative")
        or ""
    )


def pack(matches: List[Any], budget: int) -> Tuple[List[Any], int]:
    """
    Select a subset of Pinecone matches whose total token "cost"
    (as stored in metadata['tokens_estimate']) does not exceed `budget`.

    Returns (chosen_matches, used_tokens).
    """
    chosen: List[Any] = []
    used = 0

    for m in matches:
        md = getattr(m, "metadata", {}) or {}
        cost = md.get("tokens_estimate")

        # Very rough fallback if no cost recorded
        if cost is None:
            txt = _extract_text(md)
            cost = max(1, len(txt) // 4)  # ≈ tokens

        # If adding this chunk would blow the budget, skip it
        if used + cost > budget:
            continue

        chosen.append(m)
        used += cost

    return chosen, used


def render_pack(matches: List[Any]) -> str:
    """
    Turn a list of packed matches into a single text block for QA.

    Each chunk is preceded by a header like:
      [ccu p9]
    or, if page is missing:
      [ccu]
    """
    blocks: List[str] = []

    for m in matches:
        md = getattr(m, "metadata", {}) or {}
        kind = (md.get("kind") or "chunk").lower()
        page = md.get("page")

        header = f"[{kind} p{page}]" if page is not None else f"[{kind}]"
        body = _extract_text(md)

        if not body.strip():
            continue

        blocks.append(f"{header}\n{body}")

    return "\n\n".join(blocks)
