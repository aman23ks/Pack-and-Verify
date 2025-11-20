import os
from typing import List, Dict, Any

import backoff
import requests

from ..config import CONF
from ..cache import CACHE


@backoff.on_exception(backoff.expo, (requests.RequestException,), max_tries=5)
def partition_pdf(path: str) -> List[Dict[str, Any]]:
    """
    Call the Unstructured Partition API for PDFs with:
      - strategy = hi_res      (needed for layout / images / tables)
      - extract_image_block_types = ["Image", "Table"]
        ==> API returns base64 crops in metadata["image_base64"] for those elements
      - pdf_infer_table_structure = True
        ==> Table elements get metadata["text_as_html"]

    We also bump the cache key to 'uns:v2:' so old responses
    (without image_base64 / text_as_html) are ignored.
    """

    # bump version here to force fresh calls and avoid stale cache
    cache_key = f"uns:v2:{path}"
    cached = CACHE.get("uns", cache_key)
    if cached is not None:
        return cached

    filename = os.path.basename(path)
    with open(path, "rb") as f:
        files = {
            "files": (filename, f, "application/pdf"),
        }

        # These names match the official Partition Endpoint parameters
        # documented at https://docs.unstructured.io/api-reference/partition/api-parameters
        data = {
            "strategy": "hi_res",
            "split_pdf_page": True,
            # key part: ask for Image + Table blocks as base64 in metadata.image_base64
            "extract_image_block_types": ["Image", "Table"],
            # ensure tables get text_as_html
            "pdf_infer_table_structure": True,
        }

        r = requests.post(
            CONF.unstructured_url,
            headers={"UNSTRUCTURED-API-KEY": CONF.unstructured_key},
            files=files,
            data=data,
            timeout=240,
        )
        r.raise_for_status()
        out = r.json()
        CACHE.set("uns", cache_key, out)
        return out
