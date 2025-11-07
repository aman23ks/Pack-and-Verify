from typing import List, Dict, Any
from pav.clients.pinecone_index import search as pine_search

def search(query: str, top_k: int = 80):
    return pine_search(query, top_k=top_k)
