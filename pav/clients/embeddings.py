# pav/clients/embeddings.py
import logging
import os
from typing import List

import google.generativeai as genai
from dotenv import load_dotenv

logger = logging.getLogger(__name__)

# -------------------------------------------------------------
# Load .env and configure Gemini
# -------------------------------------------------------------
# This ensures that when you run python -q or cli.py, the .env
# is actually read before we look up any keys.
load_dotenv()

# Prefer GEMINI_API_KEY, but fall back to GOOGLE_API_KEY
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")

if not GEMINI_API_KEY:
    logger.warning("GEMINI_API_KEY / GOOGLE_API_KEY is not set – Gemini embeddings will fail.")
else:
    genai.configure(api_key=GEMINI_API_KEY)

# Default embedding model: text-embedding-004 (768-dim as of now)
EMBED_MODEL = os.getenv("GEMINI_EMBED_MODEL", "models/text-embedding-004")

_EMBED_DIM: int | None = None


def _probe_dim() -> int:
    """
    One-time probe to discover the embedding dimension.
    """
    global _EMBED_DIM
    if _EMBED_DIM is not None:
        return _EMBED_DIM

    if not GEMINI_API_KEY:
        raise RuntimeError("GEMINI_API_KEY / GOOGLE_API_KEY is not set – cannot probe embedding dimension.")

    logger.info("Probing Gemini embedding dimension for model %s", EMBED_MODEL)
    resp = genai.embed_content(
        model=EMBED_MODEL,
        content="dimension probe",
        task_type="retrieval_document",
    )

    # google-generativeai can return a dict or an object with .embedding
    if hasattr(resp, "embedding"):
        vec = resp.embedding
    elif isinstance(resp, dict):
        vec = resp.get("embedding")
    else:
        raise RuntimeError(f"Unexpected embed_content() response type: {type(resp)}")

    if not isinstance(vec, list) or len(vec) == 0:
        raise RuntimeError(f"Gemini returned invalid embedding during probe: {vec}")

    _EMBED_DIM = len(vec)
    logger.info("Gemini embedding dimension detected: %d", _EMBED_DIM)
    return _EMBED_DIM


def _zero_vec() -> List[float]:
    """
    Return a zero vector of the correct dimension.
    """
    dim = _probe_dim()
    return [0.0] * dim


def _embed_one(text: str) -> List[float]:
    """
    Embed a single string with Gemini and return the embedding vector.
    Guarantees a non-empty vector of fixed dimension.
    """
    if not GEMINI_API_KEY:
        raise RuntimeError("GEMINI_API_KEY / GOOGLE_API_KEY is not set – cannot embed.")

    # Normalize empty / whitespace input to a zero vector
    if not text or not text.strip():
        logger.debug("Empty text passed to embed(); returning zero vector.")
        return _zero_vec()

    resp = genai.embed_content(
        model=EMBED_MODEL,
        content=text,
        task_type="retrieval_document",
    )

    if hasattr(resp, "embedding"):
        vec = resp.embedding
    elif isinstance(resp, dict):
        vec = resp.get("embedding")
    else:
        raise RuntimeError(f"Unexpected embed_content() response type: {type(resp)}")

    if not isinstance(vec, list):
        raise RuntimeError(f"Gemini returned non-list embedding: {type(vec)}")

    if len(vec) == 0:
        # Hard fail here so you see if Gemini ever returns nonsense
        raise RuntimeError("Gemini returned an empty embedding vector (len 0).")

    dim = _probe_dim()
    if len(vec) != dim:
        # Normalize to the probed dim if something weird happens
        logger.error(
            "Embedding dim %d != probed dim %d; truncating/padding.",
            len(vec),
            dim,
        )
        if len(vec) > dim:
            vec = vec[:dim]
        else:
            vec = vec + [0.0] * (dim - len(vec))

    return vec


def embed(texts: List[str]) -> List[List[float]]:
    """
    Batch wrapper to match the interface used elsewhere in the codebase:
        vectors = embed(["text1", "text2", ...])

    Guarantees:
    - Every returned vector is a list[float] with len == EMBED_DIM.
    - No [] (dimension 0) vectors.
    """
    if not isinstance(texts, list):
        raise TypeError(f"embed() expects List[str], got {type(texts)}")

    vectors: List[List[float]] = []
    first_ok_dim: int | None = None

    for t in texts:
        try:
            vec = _embed_one(t)
            if first_ok_dim is None:
                first_ok_dim = len(vec)
            vectors.append(vec)
        except Exception as e:
            logger.error("Error embedding text %r: %s", (t or "")[:80], e)
            # If we already know the dimension, insert a zero vector to keep batch shape
            if first_ok_dim is not None:
                vectors.append([0.0] * first_ok_dim)
            else:
                # If even the first one fails, re-raise so the problem is obvious
                raise

    return vectors
