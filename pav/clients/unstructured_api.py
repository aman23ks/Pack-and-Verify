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
        data  = {"strategy": "hi_res", "split_pdf_page": True}
        r = requests.post(CONF.unstructured_url,
                          headers={"UNSTRUCTURED-API-KEY": CONF.unstructured_key},
                          files=files, data=data, timeout=240)
        r.raise_for_status()
        out = r.json()
        CACHE.set("uns", cache_key, out)
        return out
