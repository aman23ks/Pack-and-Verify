# pav/ingest/build_bundles.py

from typing import Any, Dict, List, Sequence

from .discourse_units import build_ccus


def build(elements: Sequence[Any], doc_id: str) -> List[Dict[str, Any]]:
    """
    Backwards-compatible entrypoint used by cli.py.

    Takes raw unstructured elements + doc_id and returns a flat list of
    Pinecone-ready vectors:

        {
            "id": str,
            "values": List[float],   # Gemini embedding
            "metadata": Dict[str, Any]
        }

    All the heavy lifting (CCUs + image/table children + embeddings)
    happens inside discourse_units.build_ccus().
    """
    return build_ccus(elements, doc_id)


# Optional alias if you ever want to import build_bundles elsewhere
build_bundles = build
