# pav/ingest/discourse_units.py
from typing import List, Dict, Any, Tuple
import base64

from ..types import Bundle
from ..utils import tok_len
from ..config import CONF
from ..clients.gemini import vision, contextualize

TEXT  = {"Title","NarrativeText","ListItem","Header","Footer","Paragraph"}
TABLE = {"Table"}
IMAGE = {"Image"}
HEADS = {"Title","Header"}

def _y_top(e: Dict[str,Any]) -> float:
    pts = (e.get("metadata",{}).get("coordinates",{}).get("points",[[-1,-1]]))
    return pts[0][1] if pts and len(pts[0])==2 else -1.0

def _section_indices(elements: List[Dict[str,Any]]) -> List[int]:
    starts = []
    for i,e in enumerate(elements):
        if e.get("type") in HEADS and (e.get("text") or "").strip():
            starts.append(i)
    if 0 not in starts: starts = [0] + starts
    return sorted(set(starts))

def _slice(lst, a, b):  # inclusive-exclusive
    return [(i,e) for i,e in lst[a:b]]

def _neighbor_text(arr, idx, up=3, down=3) -> List[str]:
    out=[]
    # above
    c=0
    for j in range(idx-1,-1,-1):
        ej=arr[j][1]
        if ej.get("type") in TEXT:
            t=(ej.get("text") or "").strip()
            if t: out.append(t); c+=1
            if c==up: break
    out = list(reversed(out))
    # below
    c=0
    for j in range(idx+1,len(arr)):
        ej=arr[j][1]
        if ej.get("type") in TEXT:
            t=(ej.get("text") or "").strip()
            if t: out.append(t); c+=1
            if c==down: break
    return out

def _collect_caption(e: Dict[str,Any], arr: List[Tuple[int,Dict[str,Any]]], idx: int) -> str:
    if idx-1 >= 0:
        prev = arr[idx-1][1]
        if prev.get("type") in TEXT:
            t=(prev.get("text") or "").strip()
            if t.lower().startswith(("fig", "figure", "table")) or "caption" in t.lower():
                return t
    return ""

def _render_section_text(arr: List[Tuple[int,Dict[str,Any]]]) -> Tuple[str, List[str]]:
    lines, kids = [], []
    for _,e in arr:
        if e.get("type") in TEXT:
            t=(e.get("text") or "").strip()
            if t:
                lines.append(t)
                kids.append(e.get("element_id",""))
    return "\n".join(lines).strip(), kids

def _render_tables_images(arr: List[Tuple[int,Dict[str,Any]]]) -> Tuple[str, List[str]]:
    """
    Create a readable block for all tables/images in this section, each followed by a
    faithful, context-aware narrative from 3-above/3-below neighbors.
    """
    blocks, kids = [], []
    up, down = CONF.CCU_STRUCT_NEIGHBORS

    for k,(_,e) in enumerate(arr):
        t = e.get("type")
        if t not in (TABLE | IMAGE):
            continue

        neigh_list = _neighbor_text([*arr], k, up=up, down=down)
        neighbors = "\n".join([s for s in neigh_list if s.strip()])
        caption = _collect_caption(e, [*arr], k)

        if t in TABLE:
            md = (e.get("metadata") or {})
            # Prefer HTML variants if present
            table_html = md.get("text_as_html") or md.get("text_by_html") or ""
            table_txt  = (e.get("text") or "").strip()
            primary    = table_html if table_html else table_txt

            narr = contextualize("table", primary_payload=primary, neighbors_text=neighbors, caption=caption)

            comp = "\n".join([
                (caption or "").strip(),
                "[Table]",
                table_txt if table_txt else "[no plain table text available]",
                "[Contextual narrative]",
                narr
            ]).strip()

            if comp:
                blocks.append(comp)
                kids.append(e.get("element_id",""))

        else:  # IMAGE
            md = (e.get("metadata") or {})
            b64 = md.get("image_base64")
            if not b64:
                continue
            img = base64.b64decode(b64)
            vis_text = vision(img, neighbors)
            narr = contextualize("image", primary_payload=vis_text, neighbors_text=neighbors, caption=caption)

            comp = "\n".join([
                (caption or "").strip(),
                "[Figure]",
                vis_text,
                "[Contextual narrative]",
                narr
            ]).strip()

            if comp:
                blocks.append(comp)
                kids.append(e.get("element_id",""))

    return ("\n\n".join(blocks)).strip(), kids

def _split_if_needed(text: str, children: List[str], doc_id: str, page: int) -> List[Bundle]:
    toks = tok_len(text)
    if toks <= CONF.CCU_MAX_TOKENS:
        return [{
            "id": f"{doc_id}_p{page}_ccu_{hash((text,page)) & 0xfffffff}",
            "kind": "ccu",
            "page": page,
            "text_main": text,
            "context_pre": [],
            "context_post": "",
            "cost": toks,
            "children": children
        }]
    parts = text.split("\n")
    bundles=[]
    buf=[]; count=0
    i=0
    while i < len(parts):
        p=parts[i]
        t=tok_len(p)
        if count + t > CONF.CCU_MAX_TOKENS and buf:
            chunk="\n".join(buf).strip()
            bundles.append({
                "id": f"{doc_id}_p{page}_ccu_{hash((chunk, i, page)) & 0xfffffff}",
                "kind":"ccu","page":page,"text_main":chunk,
                "context_pre":[],"context_post":"","cost": tok_len(chunk),
                "children": children
            })
            # overlap tokens
            back=len(buf)-1; back_tok=0
            while back>0 and back_tok<CONF.CCU_OVERLAP_TOKENS:
                back_tok += tok_len(buf[back]); back -= 1
            buf = buf[max(0, back):]
            count = tok_len("\n".join(buf))
        buf.append(p); count += t; i+=1
    if buf:
        chunk="\n".join(buf).strip()
        bundles.append({
            "id": f"{doc_id}_p{page}_ccu_{hash((chunk, i, page)) & 0xfffffff}",
            "kind":"ccu","page":page,"text_main":chunk,
            "context_pre":[],"context_post":"","cost": tok_len(chunk),
            "children": children
        })
    return bundles

def build_ccus(elements: List[Dict[str,Any]], doc_id: str) -> List[Bundle]:
    bundles: List[Bundle] = []
    arr = list(enumerate(elements))
    starts = _section_indices(elements)

    for si, start in enumerate(starts):
        end = starts[si+1] if si+1 < len(starts) else len(elements)
        sec_slice = _slice(arr, start, end)
        if not sec_slice:
            continue
        first_page = (sec_slice[0][1].get("metadata") or {}).get("page_number", 0)

        # Section text and media (with contextual narratives)
        text_body, kids_text = _render_section_text(sec_slice)
        tables_images, kids_media = _render_tables_images(sec_slice)

        # Simple breadcrumb (optional; could be improved later)
        heads = [ (e.get("text") or "").strip()
                  for _,e in sec_slice if e.get("type") in HEADS and (e.get("text") or "").strip() ]
        breadcrumb = " › ".join(heads) if heads else ""

        parts = []
        if breadcrumb: parts.append(f"[Section] {breadcrumb}")
        if text_body:  parts.append(text_body)
        if tables_images:
            parts.append("\n[Figures/Tables]\n" + tables_images)
        full = "\n\n".join([p for p in parts if p]).strip()
        if not full:
            continue

        kids = kids_text + kids_media
        bundles.extend(_split_if_needed(full, kids, doc_id, first_page))

    return bundles
