import os
from typing import List, Any

# Optional Gemini: if GOOGLE_API_KEY is set and google-generativeai is installed, we'll use it.
_USE_GEMINI = False
_MODEL = None
try:
    import google.generativeai as genai
    if os.environ.get("GOOGLE_API_KEY"):
        genai.configure(api_key=os.environ["GOOGLE_API_KEY"])
        _MODEL = genai.GenerativeModel("gemini-1.5-pro")
        _USE_GEMINI = True
except Exception:
    _USE_GEMINI = False

TABLE_TYPES = {"Table", "TableChunk"}
IMAGE_TYPES = {"Image"}

def _etype(el: Any) -> str:
    return getattr(el, "category", None) or getattr(el, "type", None) or getattr(el, "element_type", None) or "Element"

def _get_text(el: Any) -> str:
    txt = getattr(el, "text", None)
    if isinstance(txt, str) and txt.strip():
        return txt.strip()
    md = getattr(el, "metadata", {}) or {}
    for k in ("text", "text_as_html", "raw_text"):
        if isinstance(md.get(k), str) and md[k].strip():
            return md[k].strip()
    return ""

def _set_text(el: Any, new_text: str):
    if hasattr(el, "text"):
        el.text = new_text
    else:
        if not hasattr(el, "metadata") or el.metadata is None:
            el.metadata = {}
        el.metadata["text"] = new_text
    # mark enriched
    if not hasattr(el, "metadata") or el.metadata is None:
        el.metadata = {}
    el.metadata["contextualized"] = True

def _neighbors(elems: List[Any], i: int, k: int) -> List[Any]:
    start = max(0, i - k)
    end = min(len(elems), i + k + 1)
    return [elems[j] for j in range(start, end) if j != i]

def _neighbors_bulleted(neighs: List[Any]) -> str:
    out = []
    for n in neighs:
        t = _etype(n)
        txt = _get_text(n)
        if not txt:
            continue
        if len(txt) > 1200:
            txt = txt[:1200] + " …"
        out.append(f"- [{t}] {txt}")
    return "\n".join(out)

def _gemini_contextualize(raw_block: str, neighbor_bullets: str) -> str:
    if not _USE_GEMINI:
        return (
            "Contextualized description (fallback, no Gemini available):\n"
            + (raw_block[:1500] + (" …" if len(raw_block) > 1500 else ""))
            + ("\n\nNearby context:\n" + neighbor_bullets if neighbor_bullets else "")
        ).strip()

    sys_rules = (
        "You write faithful, concise context text for retrieval. "
        "Preserve headers/units/labels. No speculation. "
        "One paragraph, <= 1800 characters."
    )
    user_prompt = f"""
TABLE/IMAGE RAW:
{raw_block}
NEARBY CONTEXT (±3 elements):
{neighbor_bullets}

Task: Return ONE paragraph that accurately describes the figure/table with context:
- Include headers/units/footnotes when visible.
- Identify entities, dates, measures clarified by neighbors.
- Do not invent facts or analyze beyond the snippets.
"""
    resp = _MODEL.generate_content(
        [{"role": "system", "parts": [sys_rules]},
         {"role": "user", "parts": [user_prompt]}],
        safety_settings=None,
        generation_config=genai.types.GenerationConfig(
            temperature=0.2, top_p=0.9, max_output_tokens=512
        ),
    )
    text = (getattr(resp, "text", None) or "").strip()
    return text or "insufficient context"


def enrich_with_context(elements: List[Any], neighbors: int = 3) -> List[Any]:
    """
    For each Table/TableChunk/Image element:
        - collect ±neighbors elements,
        - ask Gemini (or fallback) to produce faithful contextual text,
        - replace the element's text with the contextualized text.
    Mutates and returns the same list for convenience.
    """
    for i, el in enumerate(elements):
        typ = _etype(el)
        if typ in TABLE_TYPES or typ in IMAGE_TYPES:
            raw = _get_text(el)
            neigh = _neighbors(elements, i, neighbors)
            bullets = _neighbors_bulleted(neigh)
            new_text = _gemini_contextualize(raw, bullets)
            _set_text(el, new_text)
    return elements
