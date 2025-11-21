from typing import Optional, Dict, List

from pav.clients.pinecone_index import search as pine_search


def search(
    query: str,
    top_k: int = 5,
    namespace: Optional[str] = None,
    filter: Optional[Dict] = None,
):
    """
    Thin wrapper around Pinecone search so other code can import
    pav.clients.retrieval.search.search without caring about Pinecone details.

    Parameters
    ----------
    query : str
        Natural language query.
    top_k : int
        Number of results to return.
    namespace : Optional[str]
        Pinecone namespace to search in. Should usually match the doc_id
        (PDF filename stem) that was used during ingest, e.g. "2404.07973v1".
    filter : Optional[Dict]
        Optional Pinecone metadata filter.

    Returns
    -------
    List[pinecone.grpc.Index.QueryResponse.Match]
        Matches with metadata attached.
    """
    return pine_search(
        query=query,
        top_k=top_k,
        namespace=namespace,
        filter=filter,
    )
