# pav/clients/rerank_local.py
from __future__ import annotations
import logging
from typing import Iterable, List, Any, Tuple


logger = logging.getLogger(__name__)

# ----------------------------
# Optional import (graceful fallback if missing)
# ----------------------------
try:
    from sentence_transformers import CrossEncoder
    _HAS_ST = True
except Exception as _e:
    logger.warning("sentence-transformers not available for local rerank; falling back.")
    CrossEncoder = None
    _HAS_ST = False

# Small, free cross-encoder that runs locally on CPU
_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"
_MODEL = None  # lazy-loaded
RERANK_AVAILABLE = _HAS_ST  # export this for debugging

def _extract_text(md: dict) -> str:
    """
    Mirrors the packer logic for extracting text from match metadata.
    """
    return (
        (md.get("content") or md.get("text_main") or md.get("narrative") or "")
        if isinstance(md, dict) else ""
    )

def _ensure_md(match: Any) -> dict:
    """
    Works for pinecone Match objects or dict-like results.
    """
    try:
        md = getattr(match, "metadata", None)
        if md is None and isinstance(match, dict):
            md = match.get("metadata", {})
        return md or {}
    except Exception:
        return {}

def _score_pairs_ce(query: str, texts: List[str]) -> List[float]:
    global _MODEL
    if not _HAS_ST:
        # No sentence-transformers installed → return zero scores so we don't reorder
        return [0.0] * len(texts)
    if _MODEL is None:
        # Lazy load once; defaults to CPU
        _MODEL = CrossEncoder(_MODEL_NAME)
    pairs = [(query, t) for t in texts]
    # returns higher-is-better scores
    scores = _MODEL.predict(pairs)
    return [float(s) for s in scores]


def local_rerank(query: str, matches: Iterable[Any]) -> List[Any]:
    matches = list(matches)
    if not matches:
        return matches

    texts = []
    for m in matches:
        md = _ensure_md(m)
        txt = _extract_text(md).strip()
        texts.append(txt)

    try:
        scores = _score_pairs_ce(query, texts)
    except Exception as e:
        logger.error("Local rerank failed: %s", e)
        return matches

    # attach rerank score for visibility
    for m, s in zip(matches, scores):
        md = _ensure_md(m)
        md["_rerank_score"] = float(s)

    # sort by rerank score desc
    return sorted(matches, key=lambda m: _ensure_md(m).get("_rerank_score", 0.0), reverse=True)

