from typing import Any, Dict, List, Sequence, Tuple
import math


def _estimate_tokens(text: str) -> int:
    """
    Rough token estimate assuming ~4 characters per token.
    """
    if not text:
        return 0
    return max(1, int(math.ceil(len(text) / 4.0)))


def _get_metadata(match: Any) -> Dict[str, Any]:
    """
    Safely extract metadata dict from a Pinecone match or a plain dict.
    """
    md = {}
    # Pinecone ScoredVector: .metadata attribute
    if hasattr(match, "metadata"):
        md = match.metadata or {}
    # Fallback if it's already a dict-like
    elif isinstance(match, dict):
        md = match.get("metadata") or {}
    if md is None:
        md = {}
    return md


def _get_score(match: Any) -> float:
    """
    Safely extract similarity score from a match.
    """
    if hasattr(match, "score") and match.score is not None:
        return float(match.score)
    if isinstance(match, dict) and "score" in match:
        try:
            return float(match["score"])
        except Exception:
            return 0.0
    return 0.0


def _get_vector(match: Any) -> List[float]:
    """
    Get the stored embedding vector for a match.
    Requires Pinecone query(include_values=True).
    """
    if hasattr(match, "values") and match.values is not None:
        return list(match.values)
    if isinstance(match, dict) and "values" in match and match["values"] is not None:
        return list(match["values"])
    return []


def _get_content(md: Dict[str, Any]) -> str:
    """
    Decide which text field to use for context and token estimation.
    """
    return (
        md.get("content")
        or md.get("text_main")
        or md.get("narrative")
        or ""
    )


def _get_tokens(md: Dict[str, Any], content: str) -> int:
    """
    Get token count from metadata if present, else estimate from content length.
    """
    t = md.get("tokens_estimate")
    if isinstance(t, int) and t > 0:
        return t
    return _estimate_tokens(content)


def _cosine(a: List[float], b: List[float]) -> float:
    """
    Cosine similarity between two vectors. Returns 0.0 if invalid.
    """
    if not a or not b:
        return 0.0
    if len(a) != len(b):
        # Truncate to common length if shapes differ
        n = min(len(a), len(b))
        a = a[:n]
        b = b[:n]
    dot = 0.0
    na = 0.0
    nb = 0.0
    for x, y in zip(a, b):
        dot += x * y
        na += x * x
        nb += y * y
    if na <= 0.0 or nb <= 0.0:
        return 0.0
    return dot / (math.sqrt(na) * math.sqrt(nb))


def pack(
    matches: Sequence[Any],
    budget_tokens: int,
    lambda_penalty: float = 0.3,
) -> Tuple[List[Any], int]:
    """
    Select a subset of `matches` under a token budget using a greedy
    diversity-aware objective:

        effective_score = base_score - lambda_penalty * max_cos_sim_to_selected

    where `base_score` is the Pinecone similarity score, and the penalty
    discourages adding chunks that are too redundant with already-selected ones.

    Args:
        matches: list of Pinecone matches (ScoredVector) or dicts.
        budget_tokens: maximum total tokens allowed in the packed context.
        lambda_penalty: weight for cosine-similarity redundancy penalty.

    Returns:
        (chosen_matches, used_tokens)
    """
    if budget_tokens <= 0 or not matches:
        return [], 0

    print(f"[PACK] Starting pack: {len(matches)} matches, budget={budget_tokens}, lambda_penalty={lambda_penalty}")

    # Precompute useful info for each match
    items: List[Dict[str, Any]] = []
    for idx, m in enumerate(matches):
        md = _get_metadata(m)
        content = _get_content(md).strip()
        if not content:
            # If there's literally no usable text, skip it
            continue
        tokens = _get_tokens(md, content)
        if tokens <= 0:
            continue

        score = _get_score(m)
        vec = _get_vector(m)

        items.append(
            {
                "idx": idx,           # index into original matches list
                "match": m,           # original object
                "metadata": md,
                "content": content,
                "tokens": tokens,
                "score": score,
                "vec": vec,
            }
        )

    if not items:
        print("[PACK] No items with usable content/tokens – nothing to pack.")
        return [], 0

    selected: List[Dict[str, Any]] = []
    used_tokens = 0

    # Greedy selection with redundancy penalty
    while True:
        best_item = None
        best_effective = None

        for cand in items:
            # Skip already-selected (we mark by setting "selected" flag)
            if cand.get("selected"):
                continue

            # Respect token budget
            if used_tokens + cand["tokens"] > budget_tokens:
                # Debug print for budget skip
                m_obj = cand["match"]
                mid = getattr(m_obj, "id", None)
                if mid is None and isinstance(m_obj, dict):
                    mid = m_obj.get("id")
                md = cand["metadata"]
                print(
                    f"[PACK] SKIP (budget) id={mid} "
                    f"kind={md.get('kind')} page={md.get('page')} "
                    f"cost={cand['tokens']} "
                    f"would_be={used_tokens + cand['tokens']} > budget={budget_tokens}"
                )
                continue

            base = cand["score"]

            # Compute max cosine similarity to already-selected items
            penalty_sim = 0.0
            if selected and cand["vec"]:
                sims = [
                    _cosine(cand["vec"], s["vec"])
                    for s in selected
                    if s["vec"]
                ]
                if sims:
                    penalty_sim = max(sims)

            effective = base - lambda_penalty * penalty_sim

            if best_item is None or effective > best_effective:
                best_item = cand
                best_effective = effective

        if best_item is None:
            # No more candidates fit under budget
            break

        # Mark as selected and update counters
        best_item["selected"] = True
        selected.append(best_item)
        used_tokens += best_item["tokens"]

        # Recompute debug metrics for printing
        base = best_item["score"]
        penalty_sim = 0.0
        if selected and best_item["vec"]:
            sims = [
                _cosine(best_item["vec"], s["vec"])
                for s in selected
                if s is not best_item and s["vec"]
            ]
            if sims:
                penalty_sim = max(sims)
        eff_score = base - lambda_penalty * penalty_sim

        m_obj = best_item["match"]
        mid = getattr(m_obj, "id", None)
        if mid is None and isinstance(m_obj, dict):
            mid = m_obj.get("id")
        md = best_item["metadata"]
        kind = md.get("kind")
        page = md.get("page")
        cost = best_item["tokens"]

        print(
            f"[PACK] SELECTED id={mid} "
            f"kind={kind} page={page} "
            f"cost={cost} base_score={base:.4f} "
            f"max_sim={penalty_sim:.4f} eff_score={eff_score:.4f} "
            f"cum_tokens={used_tokens}"
        )

    # Return the original match objects in the selected order
    chosen_matches = [s["match"] for s in selected]
    print(f"[PACK] Finished pack: selected={len(chosen_matches)}, used_tokens={used_tokens}")
    return chosen_matches, used_tokens


def render_pack(matches: Sequence[Any]) -> str:
    """
    Turn a packed list of matches into a single context string.
    We rely on metadata fields populated at ingest time.
    """
    blocks: List[str] = []
    for m in matches:
        md = _get_metadata(m)
        kind = (md.get("kind") or "chunk").lower()
        page = md.get("page")
        header = f"[{kind}"
        if page is not None:
            header += f" p{page}"
        header += "]"

        body = (
            md.get("content")
            or md.get("text_main")
            or md.get("narrative")
            or ""
        ).strip()

        block = header + "\n" + body
        blocks.append(block)

    return "\n\n---\n\n".join(blocks)
