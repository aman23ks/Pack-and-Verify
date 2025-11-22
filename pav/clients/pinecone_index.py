# pav/clients/pinecone_index.py

import os
from typing import Dict, List, Optional

from pinecone import Pinecone
from pav.clients.embeddings import embed

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

# Keep index naming flexible: env first, then default
PINECONE_INDEX_NAME = (
    os.getenv("PINECONE_INDEX")
    or os.getenv("PINECONE_INDEX_NAME")
    or "pav-quick"
)

if not PINECONE_API_KEY:
    raise RuntimeError("PINECONE_API_KEY is not set")

pc = Pinecone(api_key=PINECONE_API_KEY)

if not PINECONE_INDEX_NAME:
    INDEX = None
    print("PINECONE_INDEX_NAME / PINECONE_INDEX is not set – cannot create Index handle.")
else:
    INDEX = pc.Index(PINECONE_INDEX_NAME)


def _extract_primary_text(bundle: Dict) -> str:
    """
    Decide what text to embed.

    - For images/tables: prefer metadata['narrative'] if present.
    - Otherwise: fall back to bundle['text'].
    """
    md = bundle.get("metadata") or {}
    kind = (md.get("kind") or "").lower()

    if kind in ("image", "figure", "table"):
        nar = (md.get("narrative") or "").strip()
        if nar:
            return nar

    raw = (bundle.get("text") or "").strip()
    return raw


def _to_pinecone_vectors(doc_id: str, bundles: List[Dict]) -> List[Dict]:
    """
    Convert internal bundles to Pinecone vector dicts:
      { "id": ..., "values": [...], "metadata": {...} }
    """
    vectors: List[Dict] = []

    ids: List[str] = []
    metas: List[Dict] = []
    texts: List[str] = []

    for b in bundles:
        vid = b["id"]
        md = b.get("metadata", {}) or {}

        content_text = _extract_primary_text(b)
        if not content_text:
            continue

        # store the final text we actually index
        md["content"] = content_text

        ids.append(vid)
        metas.append(md)
        texts.append(content_text)

    if not texts:
        return []

    embs = embed(texts)  # list[list[float]]

    for vid, md, vals in zip(ids, metas, embs):
        if not vals or len(vals) == 0:
            continue
        vectors.append(
            {
                "id": vid,
                "values": vals,
                "metadata": md,
            }
        )

    return vectors


def upsert(doc_id: str, bundles: List[Dict]) -> None:
    """
    Upsert all vectors for a given document into Pinecone under namespace=doc_id.
    """
    if INDEX is None:
        raise RuntimeError("Pinecone index handle not configured; cannot upsert.")

    vectors = _to_pinecone_vectors(doc_id, bundles)
    if not vectors:
        print(f"[PINECONE] No vectors to upsert for {doc_id}")
        return

    batch_size = 100
    for i in range(0, len(vectors), batch_size):
        batch = vectors[i : i + batch_size]
        INDEX.upsert(vectors=batch, namespace=doc_id)

    print(f"[PINECONE] Upserted {len(vectors)} vectors for {doc_id} into namespace '{doc_id}'")


def search(
    query: str,
    top_k: int = 5,
    namespace: Optional[str] = None,
    filter: Optional[Dict] = None,
):
    """
    Simple wrapper so pav.clients.retrieval.search can import `search`.

    Returns Pinecone matches with metadata.

    NOTE:
    - `namespace` must match the doc_id you ingested with (PDF filename stem),
      e.g. "2404.07973v1".
    """
    if INDEX is None:
        raise RuntimeError("Pinecone index handle not configured; cannot search.")

    print("-----------query-----------", query)
    print("-----------namespace-----------", namespace)

    vec = embed([query])[0]
    res = INDEX.query(
        vector=vec,
        top_k=top_k,
        namespace=namespace,
        filter=filter,
        include_metadata=True,
        include_values=True,
    )

    print("-----------res matched-----------", res.matches)
    return res.matches
