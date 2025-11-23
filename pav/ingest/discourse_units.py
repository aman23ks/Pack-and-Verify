# pav/ingest/discourse_units.py

import logging
import math
import uuid
from typing import Any, Dict, List, Tuple, Sequence

from pav.clients.gemini import vision, contextualize, embed as gemini_embed

logger = logging.getLogger(__name__)

# -------------------------------------------------------------------
# Thin embed + token helpers
# -------------------------------------------------------------------


def _embed(texts: List[str]) -> List[List[float]]:
    """
    Wrapper around pav.clients.gemini.embed().

    - Accepts list[str]
    - Returns list[list[float]] of same length
    - On failure, returns list of empty vectors so ingest doesn't crash
    """
    if isinstance(texts, str):
        texts = [texts]

    if not texts:
        return []

    try:
        return gemini_embed(texts)
    except Exception as e:
        logger.error("Failed to embed texts with Gemini: %s", e)
        return [[] for _ in texts]


def _estimate_tokens(text: str) -> int:
    """
    Rough token estimate assuming ~4 characters per token.
    """
    if not text:
        return 0
    return int(math.ceil(len(text) / 4.0))


# -------------------------------------------------------------------
# Types
# -------------------------------------------------------------------


class Element:
    """
    Thin wrapper so this still works whether `elements` are dicts or objects
    from unstructured.
    """

    def __init__(self, raw: Any):
        self.raw = raw
        self.type = (
            getattr(raw, "type", None)
            or getattr(raw, "category", None)
            or (raw.get("type") if isinstance(raw, dict) else None)
        )
        self.text = getattr(raw, "text", None) or (raw.get("text", "") if isinstance(raw, dict) else "")
        self.metadata = (
            getattr(raw, "metadata", None)
            or (raw.get("metadata", {}) if isinstance(raw, dict) else {})
            or {}
        )

    @property
    def element_id(self) -> str:
        mid = self.metadata.get("element_id")
        if mid:
            return str(mid)
        # fall back to python object id if nothing else
        return str(id(self.raw))

    @property
    def page(self) -> int:
        return int(self.metadata.get("page_number", 0))


# -------------------------------------------------------------------
# Neighbor text helpers
# -------------------------------------------------------------------


def _collect_neighbor_text(
    elems: Sequence[Element],
    idx: int,
    direction: str,
    max_blocks: int = 10,
) -> str:
    """
    Collect up to `max_blocks` textual blocks above or below a given index
    within the same page slice.
    """
    if direction not in {"above", "below"}:
        raise ValueError("direction must be 'above' or 'below'")

    collected: List[str] = []
    n = len(elems)

    if direction == "above":
        r = range(idx - 1, -1, -1)
    else:
        r = range(idx + 1, n)

    for j in r:
        if len(collected) >= max_blocks:
            break
        ej = elems[j]
        # Skip other media; we only want surrounding text
        if ej.type in {"Image", "Table"}:
            continue
        txt = (ej.text or "").strip()
        if txt:
            collected.append(txt)

    if direction == "above":
        collected.reverse()

    return "\n".join(collected)


def _collect_caption(page_elems: Sequence[Element], idx: int) -> str:
    """
    Try a few strategies:
    - direct metadata caption if present
    - nearby NarrativeText that starts with 'Figure' or 'Table'
    """
    if idx < 0 or idx >= len(page_elems):
        return ""

    elem = page_elems[idx]
    cap = elem.metadata.get("caption")
    if cap:
        return str(cap).strip()

    # look in following few text blocks
    for j in range(idx + 1, min(idx + 6, len(page_elems))):
        e = page_elems[j]
        if e.type == "NarrativeText":
            t = (e.text or "").strip()
            tl = t.lower()
            if tl.startswith("figure ") or tl.startswith("fig.") or tl.startswith("table "):
                return t
    return ""


# -------------------------------------------------------------------
# Tables / images handling
# -------------------------------------------------------------------

CCU_CONTENT_MAX = 16000
IMAGE_META_MAX = 16000
TABLE_META_MAX = 16000
CONTEXT_META_MAX = 4000
HTML_META_MAX = 4000


def _render_tables_images(
    page_elems: Sequence[Element],
    doc_id: str,
    page: int,
    parent_ccu_id: str,
) -> Tuple[List[Dict[str, Any]], List[str]]:
    """
    For a given page slice, run Gemini on every Image/Table and return:
      - list of vector dicts
      - list of child IDs (for 'children' metadata on CCU)
    """
    media_vectors: List[Dict[str, Any]] = []
    child_ids: List[str] = []

    for idx, e in enumerate(page_elems):
        if e.type not in {"Image", "Table"}:
            continue

        eid = e.element_id
        caption = _collect_caption(page_elems, idx)
        text_above = _collect_neighbor_text(page_elems, idx, "above", max_blocks=10)
        text_below = _collect_neighbor_text(page_elems, idx, "below", max_blocks=10)

        print("\n==============================")
        print(f"[CCU] Processing {e.type} element_id={eid} page={page}")
        print(f"Caption: {caption}")
        print("-- Text ABOVE (up to 10 blocks) --")
        print(f"  > {text_above or '[none]'}")
        print("-- Text BELOW (up to 10 blocks) --")
        print(f"  > {text_below or '[none]'}")
        print("================================")

        # ---------------- IMAGE ----------------
        if e.type == "Image":
            img_b64 = e.metadata.get("image_base64") or ""
            img_mime = e.metadata.get("image_mime_type") or "image/png"

            if not img_b64:
                logger.warning("Image element %s page=%s has no base64; skipping.", eid, page)
                continue

            # NOTE: we don't store the full base64 in metadata, only a small stub in HTML
            img_html = f'<img src="data:{img_mime};base64,{img_b64[:32]}..." alt="{caption}"/>'

            print("\n[IMAGE] Got base64 image. Calling Gemini.vision(...) with above context.")
            vis_prompt = (
                "You are parsing a figure from a scientific paper. Return a structured description\n"
                "of the image, focusing on chart/table structure, axes, legends, and numeric values.\n"
                "JSON is preferred when natural; otherwise return a detailed plain-text description."
            )
            vis_summary = vision(img_b64, img_mime, prompt=vis_prompt) or ""

            # FULL vision output
            print("[IMAGE] vision() FULL summary:")
            print(vis_summary or "[none]")

            print("\n[IMAGE] Calling Gemini.contextualize(kind='image', ...)")
            narr = contextualize(
                kind="image",
                text_html=img_html,
                text_above=text_above,
                text_below=text_below,
                vision_summary=vis_summary,
                doc_meta={"page": page, "element_id": eid},
            )

            # FULL narrative output
            print("[IMAGE] Gemini narrative FULL:")
            print(narr or "[none]")
            print("-------- end image block --------\n")

            child_id = f"{doc_id}_p{page}_img_{eid}"
            child_ids.append(child_id)

            text_main = narr or ""
            tokens_estimate = _estimate_tokens(text_main)

            narr_for_meta = text_main[:IMAGE_META_MAX]
            text_above_meta = (text_above or "")[:CONTEXT_META_MAX]
            text_below_meta = (text_below or "")[:CONTEXT_META_MAX]
            html_meta = img_html[:HTML_META_MAX]
            vis_meta = (vis_summary or "")[:IMAGE_META_MAX]

            vec = {
                "id": child_id,
                "values": _embed([text_main])[0],
                # cli.py expects these at top-level
                "text_main": text_main,
                "cost": tokens_estimate,
                "metadata": {
                    "doc_id": doc_id,
                    "page": page,
                    "kind": "image",
                    "element_id": eid,
                    "parent_ccu": parent_ccu_id,
                    "caption": caption or "",
                    # full rich content, all as strings for Pinecone
                    "narrative": narr_for_meta,
                    "text_above": text_above_meta,
                    "text_below": text_below_meta,
                    "html": html_meta,
                    "vision_summary": vis_meta,
                    "tokens_estimate": tokens_estimate,
                },
            }
            media_vectors.append(vec)

        # ---------------- TABLE ----------------
        elif e.type == "Table":
            table_html = e.metadata.get("text_as_html") or (e.text or "")
            table_html = (table_html or "").strip()

            print("\n[TABLE] Calling Gemini.contextualize(kind='table', ...)")
            narr = contextualize(
                kind="table",
                text_html=table_html,
                text_above=text_above,
                text_below=text_below,
                vision_summary=None,
                doc_meta={"page": page, "element_id": eid},
            )

            # FULL narrative output
            print("[TABLE] Gemini narrative FULL:")
            print(narr or "[none]")
            print("-------- end table block --------\n")

            child_id = f"{doc_id}_p{page}_tbl_{eid}"
            child_ids.append(child_id)

            text_main = narr or ""
            tokens_estimate = _estimate_tokens(text_main)

            narr_for_meta = text_main[:TABLE_META_MAX]
            text_above_meta = (text_above or "")[:CONTEXT_META_MAX]
            text_below_meta = (text_below or "")[:CONTEXT_META_MAX]
            html_meta = table_html[:HTML_META_MAX]

            vec = {
                "id": child_id,
                "values": _embed([text_main])[0],
                "text_main": text_main,
                "cost": tokens_estimate,
                "metadata": {
                    "doc_id": doc_id,
                    "page": page,
                    "kind": "table",
                    "element_id": eid,
                    "parent_ccu": parent_ccu_id,
                    "caption": caption or "",
                    "narrative": narr_for_meta,
                    "text_above": text_above_meta,
                    "text_below": text_below_meta,
                    "html": html_meta,
                    "tokens_estimate": tokens_estimate,
                },
            }
            media_vectors.append(vec)

    return media_vectors, child_ids


# -------------------------------------------------------------------
# Main: build CCUs
# -------------------------------------------------------------------


def build_ccus(elements: Sequence[Any], doc_id: str) -> List[Dict[str, Any]]:
    """
    Turn raw PDF elements into a flat list of Pinecone-ready vectors
    (CCUs + image/table child nodes).

    Returns: List[{"id": ..., "values": [...], "metadata": {...}, "text_main": "...", "cost": ...}]
    """
    # Wrap into Element objects for uniform access
    wrapped: List[Element] = [Element(e) for e in elements]

    # Group by page
    pages: Dict[int, List[Element]] = {}
    for e in wrapped:
        pages.setdefault(e.page, []).append(e)

    all_vectors: List[Dict[str, Any]] = []

    for page in sorted(pages.keys()):
        page_elems = pages[page]

        # Collect all non-media text on this page
        text_parts: List[str] = []
        for e in page_elems:
            if e.type in {"Image", "Table"}:
                continue
            t = (e.text or "").strip()
            if t:
                text_parts.append(t)

        ccu_text = "\n\n".join(text_parts).strip()
        if not ccu_text:
            ccu_text = ""
            tokens_estimate = 0
        else:
            tokens_estimate = _estimate_tokens(ccu_text)

        # Always create a CCU for structural consistency
        ccu_suffix = uuid.uuid4().hex[:8]
        ccu_id = f"{doc_id}_p{page}_ccu_{ccu_suffix}"

        # Process images/tables for this page
        media_vecs, child_ids = _render_tables_images(
            page_elems=page_elems,
            doc_id=doc_id,
            page=page,
            parent_ccu_id=ccu_id,
        )

        # Store CCU content in metadata as well (truncated for safety)
        content_meta = (ccu_text or "")[:CCU_CONTENT_MAX]

        text_main = ccu_text or ""

        ccu_vec = {
            "id": ccu_id,
            "values": _embed([text_main])[0],
            "text_main": text_main,
            # cli.py expects 'cost'
            "cost": tokens_estimate,
            "metadata": {
                "doc_id": doc_id,
                "page": page,
                "kind": "ccu",
                "tokens_estimate": tokens_estimate,
                "children": child_ids,
                "content": content_meta,
            },
        }

        print(f"\n[CCU] Built CCU for doc_id={doc_id} page={page}")
        print("  id:", ccu_id)
        print("  tokens_estimate:", tokens_estimate)
        print("  children:", child_ids)

        all_vectors.append(ccu_vec)
        all_vectors.extend(media_vecs)

    return all_vectors
