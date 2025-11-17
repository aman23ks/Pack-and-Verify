import requests, backoff
from typing import List, Dict, Any
from ..config import CONF
from ..cache import CACHE

@backoff.on_exception(backoff.expo, (requests.RequestException,), max_tries=5)
def partition_pdf(path: str) -> List[Dict[str, Any]]:
    cache_key = f"uns:{path}"
    cached = CACHE.get("uns", cache_key)
    if cached: return cached

    with open(path, "rb") as f:
        files = {"files": (path.split("/")[-1], f, "application/pdf")}
        data  = {"strategy": "hi_res", "chunking_strategy": "basic"}
        r = requests.post(CONF.unstructured_url,
                          headers={"UNSTRUCTURED-API-KEY": CONF.unstructured_key},
                          files=files, data=data, timeout=240)
        r.raise_for_status()
        out = r.json()
        filtered = [
            el for el in out
            if el.get("type") in ("CompositeElement", "Table", "Image")
        ]
        CACHE.set("uns", cache_key, filtered)
        return filtered
    
def enrich_with_context(elements, context_window=3):
    enriched = []

    for i, el in enumerate(elements):
        if el.category in ("Table", "Image"):
            # Grab nearby text context
            above = [
                e.text for e in elements[max(0, i - context_window):i]
                if hasattr(e, "text")
            ]
            below = [
                e.text for e in elements[i + 1:i + 1 + context_window]
                if hasattr(e, "text")
            ]
            el.combined_text = "\n".join(above + [el.text] + below)
            enriched.append(el)
        else:
            enriched.append(el)

    return enriched

