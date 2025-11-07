import base64
from typing import List, Dict, Any
from collections import defaultdict
from ..utils import tok_len
from ..types import Bundle
from ..clients.gemini import vision

TEXT = {"Title","NarrativeText","ListItem","Header","Footer","Paragraph"}
TABLE = {"Table"}
IMAGE = {"Image"}

def _y_top(e: Dict[str, Any]) -> float:
    return (e.get("metadata",{}).get("coordinates",{}).get("points",[[-1,-1]])[0][1])

def _by_page(elements: List[Dict[str, Any]]):
    pages = defaultdict(list)
    for i,e in enumerate(elements):
        p = (e.get("metadata") or {}).get("page_number", 0)
        pages[p].append((i,e))
    for p in pages: pages[p].sort(key=lambda t: _y_top(t[1]))
    return pages

def _neighbors(arr, idx, up=3, down=1):
    above, below = [], []
    for j in range(idx-1, -1, -1):
        ej=arr[j][1]
        if ej.get("type") in TEXT:
            t=(ej.get("text") or "").strip()
            if t: above.append(t)
            if len(above)==up: break
    for j in range(idx+1, len(arr)):
        ej=arr[j][1]
        if ej.get("type") in TEXT:
            t=(ej.get("text") or "").strip()
            if t: below.append(t)
            if len(below)==down: break
    return list(reversed(above)), (below[0] if below else "")

def build(elements: List[Dict[str, Any]], doc_id: str) -> List[Bundle]:
    bundles: List[Bundle] = []
    for page, arr in _by_page(elements).items():
        for k,(i,e) in enumerate(arr):
            t = e.get("type")
            if t in TEXT:
                txt=(e.get("text") or "").strip()
                if not txt: continue
                bundles.append({"id": f"{doc_id}_p{page}_txt_{i}",
                                "kind":"text","page":page,"text_main":txt,
                                "context_pre":[],"context_post":"","cost": tok_len(txt)})
            elif t in TABLE:
                txt=(e.get("text") or "").strip()
                pre,post=_neighbors(arr,k)
                full="\n".join([*pre, txt, post]).strip()
                bundles.append({"id": f"{doc_id}_p{page}_tbl_{i}",
                                "kind":"table","page":page,"text_main":txt,
                                "context_pre":pre,"context_post":post,"cost": tok_len(full)})
            elif t in IMAGE:
                pre,post=_neighbors(arr,k)
                b64=(e.get("metadata",{}).get("image_base64"))
                if not b64: continue
                img=base64.b64decode(b64)
                vtxt=vision(img, "\n".join([*pre, post]))
                full="\n".join([*pre, vtxt, post]).strip()
                bundles.append({"id": f"{doc_id}_p{page}_img_{i}",
                                "kind":"image+context","page":page,"text_main":vtxt,
                                "context_pre":pre,"context_post":post,"cost": tok_len(full)})
    return bundles
