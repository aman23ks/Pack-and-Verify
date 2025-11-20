# pav/clients/pinecone_index.py

import logging
import os
from typing import Any, Dict, Iterable, List, Optional

from pinecone import Pinecone, ServerlessSpec

from pav.clients.embeddings import embed as _embed

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------
# Pinecone client + index init
# ---------------------------------------------------------------------

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_CLOUD = os.getenv("PINECONE_CLOUD", "aws")
PINECONE_REGION = os.getenv("PINECONE_REGION", "us-east-1")

# Support both names; prefer PINECONE_INDEX if present
INDEX_NAME = os.getenv("PINECONE_INDEX") or os.getenv("PINECONE_INDEX_NAME")

INDEX = None

if not PINECONE_API_KEY:
    logger.error("PINECONE_API_KEY is not set – Pinecone operations will fail.")
else:
    pc = Pinecone(api_key=PINECONE_API_KEY)

    if not INDEX_NAME:
        logger.error(
            "PINECONE_INDEX / PINECONE_INDEX_NAME is not set – "
            "Pinecone index handle will not be initialized."
        )
    else:
        # Gemini text-embedding-004 is 768-dim
        dim = 768

        try:
            existing_names = pc.list_indexes().names()
        except Exception as e:
            logger.error("Failed to list Pinecone indexes: %s", e)
            existing_names = []

        if INDEX_NAME not in existing_names:
            logger.info(
                "Creating Pinecone index '%s' (dim=%d, cloud=%s, region=%s)...",
                INDEX_NAME,
                dim,
                PINECONE_CLOUD,
                PINECONE_REGION,
            )
            try:
                pc.create_index(
                    name=INDEX_NAME,
                    dimension=dim,
                    metric="cosine",
                    spec=ServerlessSpec(
                        cloud=PINECONE_CLOUD,
                        region=PINECONE_REGION,
                    ),
                )
            except Exception as e:
                logger.error("Failed to create Pinecone index '%s': %s", INDEX_NAME, e)

        try:
            INDEX = pc.Index(INDEX_NAME)
        except Exception as e:
            logger.error("Failed to get Pinecone index handle '%s': %s", INDEX_NAME, e)
            INDEX = None


# ---------------------------------------------------------------------
# Vector normalization
# ---------------------------------------------------------------------


def _normalize_vector(vec: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Take an arbitrary 'vector-like' dict and return one that Pinecone accepts.

    Allowed top-level keys: id, values, metadata, sparse_values.

    Behaviour:
    - If 'values' is missing or empty, we build an embedding from whatever
      text we can find (text, text_main, metadata.content, metadata.narrative,
      metadata.caption, metadata.vision_summary, metadata.html).
    - If we still have no usable text, return None (caller will skip).
    """
    vid = vec.get("id")
    if not vid:
        logger.warning("Skipping vector without 'id': %r", vec)
        return None

    values = vec.get("values")
    has_values = isinstance(values, list) and len(values) > 0

    if not has_values:
        # Collect text candidates
        text_candidates: List[str] = []

        t = vec.get("text")
        if isinstance(t, str):
            text_candidates.append(t)

        t = vec.get("text_main")
        if isinstance(t, str):
            text_candidates.append(t)

        meta = vec.get("metadata") or {}
        if isinstance(meta, dict):
            for key in ("content", "narrative", "caption", "vision_summary", "html"):
                mt = meta.get(key)
                if isinstance(mt, str):
                    text_candidates.append(mt)

        # Join all available text
        joined = "\n\n".join(
            s.strip() for s in text_candidates if isinstance(s, str) and s.strip()
        )

        if not joined:
            logger.warning("No text to embed for id=%s; skipping vector.", vid)
            return None

        try:
            values = _embed([joined])[0]
            has_values = True
        except Exception as e:
            logger.error("Embedding failed for id=%s: %s", vid, e)
            return None

    if not has_values:
        return None

    # Build a clean dict with only the keys Pinecone accepts
    cleaned: Dict[str, Any] = {
        "id": vid,
        "values": values,
        "metadata": vec.get("metadata") or {},
    }

    if "sparse_values" in vec:
        cleaned["sparse_values"] = vec["sparse_values"]

    return cleaned


# ---------------------------------------------------------------------
# Upsert
# ---------------------------------------------------------------------


def upsert(doc_id: str, vectors: Iterable[Dict[str, Any]], batch_size: int = 100) -> None:
    """
    Upsert an iterable of vector dicts into Pinecone under namespace=doc_id.

    Accepts both:
    - fully-formed Pinecone vectors (id, values, metadata), or
    - IR-style dicts with 'id' and 'text'/metadata only; we will embed.
    """
    if INDEX is None:
        logger.error("Pinecone index handle is not initialized; cannot upsert.")
        return

    batch: List[Dict[str, Any]] = []
    total = 0

    for vec in vectors:
        norm = _normalize_vector(vec)
        if norm is None:
            continue

        batch.append(norm)

        if len(batch) >= batch_size:
            INDEX.upsert(vectors=batch, namespace=doc_id)
            total += len(batch)
            logger.info("Upserted %d vectors so far into namespace=%s", total, doc_id)
            batch = []

    if batch:
        INDEX.upsert(vectors=batch, namespace=doc_id)
        total += len(batch)
        logger.info("Upserted %d vectors total into namespace=%s", total, doc_id)


# ---------------------------------------------------------------------
# Simple search wrapper (used by retrieval)
# ---------------------------------------------------------------------


def search(
    doc_id: str,
    query_vec: List[float],
    top_k: int = 5,
    include_metadata: bool = True,
) -> List[Dict[str, Any]]:
    """
    Query Pinecone with a vector and return matches as simple dicts.
    """
    if INDEX is None:
        logger.error("Pinecone index handle is not initialized; cannot search.")
        return []

    res = INDEX.query(
        namespace=doc_id,
        vector=query_vec,
        top_k=top_k,
        include_metadata=include_metadata,
    )

    matches_out: List[Dict[str, Any]] = []
    for m in res.matches:
        matches_out.append(
            {
                "id": m.id,
                "score": m.score,
                "metadata": getattr(m, "metadata", {}) or {},
            }
        )

    return matches_out
