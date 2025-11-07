from typing import List, Dict, Any
from pinecone import Pinecone, ServerlessSpec
from ..config import CONF
from ..clients.gemini import embed

_pc = Pinecone(api_key=CONF.pinecone_key)
if CONF.pine_index not in [i.name for i in _pc.list_indexes()]:
    _pc.create_index(
        name=CONF.pine_index, dimension=768, metric="cosine",
        spec=ServerlessSpec(cloud=CONF.pine_cloud, region=CONF.pine_region))
INDEX = _pc.Index(CONF.pine_index)

def upsert(doc_id: str, vecs: List[Dict[str, Any]]):
    # vecs: [{"id":..., "text":..., "metadata":{...}}]
    batch = []
    for v in vecs:
        emb = embed([v["text"]])[0]
        batch.append({"id": v["id"], "values": emb, "metadata": v["metadata"]})
    if batch:
        INDEX.upsert(vectors=batch)

def search(query: str, top_k: int = 60):
    v = embed([query])[0]
    return INDEX.query(vector=v, top_k=top_k, include_metadata=True).matches
