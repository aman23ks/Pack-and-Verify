# pav/clients/embeddings.py

import logging
import os
from typing import List

from openai import OpenAI

logger = logging.getLogger(__name__)

# -------------------------------------------------------------
# OpenAI setup
# -------------------------------------------------------------

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    logger.warning("OPENAI_API_KEY is not set – OpenAI embeddings will fail.")

client = OpenAI(api_key=OPENAI_API_KEY)

# Choose your embedding model
# text-embedding-3-small (1536 dim) is cheap and fine for RAG
EMBED_MODEL = os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-small")


def embed(texts: List[str]) -> List[List[float]]:
    """
    Batch wrapper to match the interface used elsewhere in the codebase:
        vectors = embed(["text1", "text2", ...])

    Returns:
      List[List[float]]  # one embedding per input string
    """
    if not isinstance(texts, list):
        raise TypeError(f"embed() expects List[str], got {type(texts)}")

    if not OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY is not set – cannot embed with OpenAI.")

    vectors: List[List[float]] = []

    for t in texts:
        t = t or ""
        if not t.strip():
            # Let pinecone_index.py drop empty vectors
            vectors.append([])
            continue

        try:
            resp = client.embeddings.create(
                model=EMBED_MODEL,
                input=t,
            )
            vec = resp.data[0].embedding
            if not isinstance(vec, list) or len(vec) == 0:
                raise RuntimeError("OpenAI returned empty embedding.")
            vectors.append(vec)
        except Exception as e:
            logger.error("Error embedding text %r: %s", t[:80], e)
            vectors.append([])

    return vectors
