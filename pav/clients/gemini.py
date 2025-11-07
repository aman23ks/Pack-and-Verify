# pav/clients/gemini.py
import base64
import json
from typing import List, Dict, Any, Optional

import backoff
import google.generativeai as genai

from ..config import CONF
from ..cache import CACHE

# ----------------------------
# Model setup
# ----------------------------
genai.configure(api_key=CONF.google_key)

def _prefs(primary: str, fallbacks: tuple) -> List[str]:
    return [primary] + [m for m in fallbacks if m != primary]

# Common generation configs
_GENCFG_TEXT = {
    "temperature": 0.0,
    "top_p": 1.0,
    "top_k": 40,
    "max_output_tokens": 2048,
}

_GENCFG_JSON = {
    "temperature": 0.0,
    "top_p": 1.0,
    "top_k": 40,
    "max_output_tokens": 4096,
    "response_mime_type": "application/json",
}

_SAFETY = {
    "HARASSMENT": "block_none",
    "HATE_SPEECH": "block_none",
    "SEXUAL": "block_none",
    "DANGEROUS": "block_none",
}

# ----------------------------
# Helper to build Gemini contents correctly
# ----------------------------
def _content_from_texts(*texts: str) -> list:
    """Return a single-user Content with provided text parts."""
    parts = [{"text": t} for t in texts if t]
    return [{"role": "user", "parts": parts}]

def _content_with_image(text: str, image_bytes: bytes) -> list:
    """Single-user Content with text + inline image Part."""
    return [{
        "role": "user",
        "parts": [
            {"text": text},
            {"inline_data": {"mime_type": "image/png",
                             "data": base64.b64encode(image_bytes).decode()}}
        ]
    }]

# ----------------------------
# Embeddings
# ----------------------------
@backoff.on_exception(backoff.expo, Exception, max_tries=5)
def embed(texts: List[str]) -> List[List[float]]:
    out: List[List[float]] = []
    for t in texts:
        c = CACHE.get("embed", t)
        if c:
            out.append(c)
            continue
        emb = genai.embed_content(model=CONF.embed_model, content=t)["embedding"]
        CACHE.set("embed", t, emb)
        out.append(emb)
    return out

# ----------------------------
# Vision: prompts (structured + summary)
# ----------------------------
def _vision_structured_prompt(context: str, doc_meta: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    meta_str = ""
    if doc_meta:
        try:
            meta_str = json.dumps(doc_meta, ensure_ascii=False)
        except Exception:
            meta_str = str(doc_meta)

    system = (
        "You are an expert scientific data extractor. Convert the figure into a faithful, compact JSON record. "
        "Do not guess. If a value is not visible or is ambiguous, set it to null and explain in `notes`."
    )

    user = f"""
PRINCIPLES
- Faithfulness over completeness. No invented values, labels, or units.
- Keep numeric strings as printed (e.g., "3.2 ± 0.5") and optionally normalize in parallel fields.
- Respect units exactly; never silently convert.
- Capture uncertainty (CI/SD/SE), sample sizes (n), significance tests.
- Capture legend/group mappings and panel labels (A/B/C…).
- If axes are unlabeled or cropped, say so in notes.
- If nearby text contradicts the image, favor the image and record the conflict.

INPUTS
- context (nearest paragraphs/captions): <<CONTEXT>>
{context[:2000]}

- doc_meta (optional JSON): <<DOC_META>>
{meta_str}

OUTPUT
Return ONE JSON object exactly matching this schema:

{{
  "doc_id": string | null,
  "page": number | null,
  "figure_id": string | null,
  "title_or_caption": string | null,
  "panels": [{{ "panel_label": string | null, "description": string | null }}],
  "plot_type": string | null,
  "axes": {{
    "x": {{"label": string | null, "unit": string | null, "scale": "linear"|"log"|"symlog"|null}},
    "y": {{"label": string | null, "unit": string | null, "scale": "linear"|"log"|"symlog"|null}},
    "secondary_y": {{"label": string | null, "unit": string | null}} | null
  }},
  "legend": [{{"key": string, "label": string}}],
  "groups": [{{"name": string, "n": number | null}}],
  "measures": [
    {{
      "what": string,
      "group": string | null,
      "value": number | null,
      "unit": string | null,
      "ci_or_sd": {{"type": "CI"|"SD"|"SE"|null, "lower": number|null, "upper": number|null, "value": string|null}},
      "p_value": string | null
    }}
  ],
  "text_verbatim": string | null,
  "notes": {{
    "unlabeled_axes": boolean | null,
    "cropped_or_unclear": boolean | null,
    "occlusions": string | null,
    "conflicts_with_context": string | null,
    "other": string | null
  }}
}}
"""
    return {"system": system, "user": user}

def _vision_summary_from_json(rec: Dict[str, Any]) -> str:
    parts = []
    cap = rec.get("title_or_caption") or rec.get("figure_id")
    if cap:
        parts.append(f"{cap}:")
    plot = rec.get("plot_type")
    if plot:
        parts.append(f"Plot type: {plot}.")
    axes = rec.get("axes") or {}
    axbits = []
    x = axes.get("x") or {}
    y = axes.get("y") or {}
    if x.get("label"):
        axbits.append(f'X={x.get("label")} ({x.get("unit") or "-"})')
    if y.get("label"):
        axbits.append(f'Y={y.get("label")} ({y.get("unit") or "-"})')
    if axbits:
        parts.append("Axes: " + "; ".join(axbits) + ".")
    legend = rec.get("legend") or []
    if legend:
        parts.append("Legend: " + ", ".join([f'{d.get("key")}:{d.get("label")}' for d in legend]) + ".")
    groups = rec.get("groups") or []
    if groups:
        parts.append("Groups: " + ", ".join([f'{g.get("name")} (n={g.get("n")})' for g in groups]) + ".")
    measures = rec.get("measures") or []
    bullets = []
    for m in measures[:20]:
        what = m.get("what") or "measure"
        grp  = m.get("group")
        val  = m.get("value")
        unit = m.get("unit")
        ci   = m.get("ci_or_sd") or {}
        pval = m.get("p_value")
        b = f"{what}"
        if grp: b += f" [{grp}]"
        if val is not None: b += f": {val}"
        if unit: b += f" {unit}"
        if ci.get("type"):
            bounds = []
            if ci.get("lower") is not None and ci.get("upper") is not None:
                bounds.append(f'[{ci["lower"]}, {ci["upper"]}]')
            elif ci.get("value"):
                bounds.append(ci["value"])
            if bounds:
                b += f" ({ci['type']} {', '.join(bounds)})"
        if pval:
            b += f", {pval}"
        bullets.append(b)
    text = " ".join(parts).strip()
    if bullets:
        text += ("\n- " + "\n- ".join(bullets))
    if not text:
        text = "[No readable figure content.]"
    return text

# ----------------------------
# Vision: structured extractor + legacy summary
# ----------------------------
@backoff.on_exception(backoff.expo, Exception, max_tries=5)
def vision_structured(image_bytes: bytes, context: str, doc_meta: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    cache_key = f"vision_json:{CONF.vision_prefs[0]}:{hash(image_bytes)}:{hash(context[:1024])}"
    cached = CACHE.get("vision_json", cache_key)
    if cached:
        return cached

    prompt = _vision_structured_prompt(context, doc_meta)
    combined = f"{prompt['system']}\n\n{prompt['user']}"
    contents = _content_with_image(combined, image_bytes)

    last_err = None
    for m in _prefs(CONF.vision_prefs[0], CONF.vision_prefs):
        try:
            resp = genai.GenerativeModel(m).generate_content(
                contents, generation_config=_GENCFG_JSON, safety_settings=_SAFETY
            )
            raw = (resp.text or "").strip()
            data = json.loads(raw) if raw else {}
            if isinstance(data, dict):
                CACHE.set("vision_json", cache_key, data)
                return data
        except Exception as e:
            last_err = e
            continue
    raise last_err or RuntimeError("Gemini vision_structured failed")

@backoff.on_exception(backoff.expo, Exception, max_tries=5)
def vision(image_bytes: bytes, context: str) -> str:
    try:
        rec = vision_structured(image_bytes, context, doc_meta=None)
        return _vision_summary_from_json(rec)
    except Exception:
        prompt = (
            "Extract faithful text/labels/numbers/units from the image. "
            "Use nearby context ONLY to disambiguate; do not invent.\n\n"
            f"Nearby context:\n{context[:2000]}"
        )
        contents = _content_with_image(prompt, image_bytes)
        last = None
        for m in _prefs(CONF.vision_prefs[0], CONF.vision_prefs):
            try:
                key = f"vision:{m}:{hash(image_bytes)}:{hash(context[:512])}"
                c = CACHE.get("vision", key)
                if c:
                    return c
                resp = genai.GenerativeModel(m).generate_content(
                    contents, generation_config=_GENCFG_TEXT, safety_settings=_SAFETY
                )
                text = (resp.text or "").strip()
                if text:
                    CACHE.set("vision", key, text)
                    return text
            except Exception as e:
                last = e
                continue
        raise last or RuntimeError("Gemini vision failed")

# ----------------------------
# Tabular extraction → tidy JSON
# ----------------------------
def _table_prompt(table_text: str, context: str, doc_meta: Optional[Dict[str, Any]] = None) -> Dict[str, str]:
    meta_str = ""
    if doc_meta:
        try:
            meta_str = json.dumps(doc_meta, ensure_ascii=False)
        except Exception:
            meta_str = str(doc_meta)

    system = (
        "You are an expert table normalizer. Convert scientific tables into tidy JSON. "
        "Do not infer values or units not shown. Preserve footnotes and missing values."
    )
    user = f"""
INPUTS
- table_text (as parsed by a PDF parser): <<TABLE>>
{table_text[:8000]}

- nearby context (caption/paragraphs): <<CONTEXT>>
{context[:2000]}

- doc_meta (optional JSON): <<DOC_META>>
{meta_str}

OUTPUT
Return ONE JSON object:

{{
  "doc_id": string | null,
  "page": number | null,
  "table_id": string | null,
  "title_or_caption": string | null,
  "columns": [{{"name": string, "unit": string | null, "original_header": string}}],
  "types": {{"col_name": "number"|"integer"|"string"|"percent"|"date"|"categorical"}},
  "footnotes": [string],
  "rows": [
    {{
      "_row_index": number,
      "cells": {{"col_name": {{
         "raw": string,
         "parsed": number|string|null,
         "unit": string|null,
         "flags": ["missing"|"approx"|"lt"|"gt"|"range"|"note"],
         "note": string|null
      }}}}
    }}
  ],
  "notes": {{
    "header_structure": "flat"|"multirow"|"unknown",
    "merged_cells_strategy": "forward_fill"|"as_is",
    "missing_token_set": [string],
    "unit_inference_source": "header"|"context"|"none",
    "other": string|null
  }}
}}
RULES
- Extract column units from headers when present; else from context only if explicit.
- Keep original cell strings in `raw`. If parsing fails, set `parsed`=null and explain via flags/note.
- Temperature 0. Output ONLY the JSON.
"""
    return {"system": system, "user": user}

@backoff.on_exception(backoff.expo, Exception, max_tries=5)
def table_structured(table_text: str, context: str, doc_meta: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    key = f"table_json:{hash(table_text)}:{hash(context[:1024])}"
    cached = CACHE.get("table_json", key)
    if cached:
        return cached

    p = _table_prompt(table_text, context, doc_meta)
    combined = f"{p['system']}\n\n{p['user']}"
    contents = _content_from_texts(combined)

    last_err = None
    for m in _prefs(CONF.text_prefs[0], CONF.text_prefs):
        try:
            resp = genai.GenerativeModel(m).generate_content(
                contents, generation_config=_GENCFG_JSON, safety_settings=_SAFETY
            )
            raw = (resp.text or "").strip()
            data = json.loads(raw) if raw else {}
            if isinstance(data, dict):
                CACHE.set("table_json", key, data)
                return data
        except Exception as e:
            last_err = e
            continue
    raise last_err or RuntimeError("Gemini table_structured failed")

# ----------------------------
# Contextualizer (tables/images → narrative)
# ----------------------------
@backoff.on_exception(backoff.expo, Exception, max_tries=5)
def contextualize(
    kind: str,
    primary_payload: str,
    neighbors_text: str,
    caption: str = "",
    doc_meta: Optional[Dict[str, Any]] = None,
) -> str:
    """
    Create a faithful narrative (<= 180 words) that explains a TABLE or FIGURE
    using its immediate 3-above/3-below neighbors.
    Returns 'Insufficient context' if inputs are too thin.
    """
    meta_str = ""
    if doc_meta:
        try:
            meta_str = json.dumps(doc_meta, ensure_ascii=False)
        except Exception:
            meta_str = str(doc_meta)

    cache_key = f"ctx_narr:{kind}:{hash(primary_payload)}:{hash(neighbors_text)}:{hash(caption)}"
    c = CACHE.get("ctx_narr", cache_key)
    if c:
        return c

    system = (
        "You are a precise scientific writer. Write one short paragraph (<= 180 words) that faithfully explains the item "
        "using ONLY the provided content. Do not guess or generalize beyond the inputs. Prefer exact numbers/units that appear. "
        "If the content is insufficient to explain the item, respond exactly with 'Insufficient context'."
    )
    user = f"""
ITEM_KIND: {kind.upper()}

CAPTION (optional):
{caption or '[none]'}

PRIMARY_PAYLOAD (verbatim; table HTML/text or figure JSON/text):
<<<PRIMARY>>>
{primary_payload[:12000]}
<<<END PRIMARY>>>

NEIGHBORS (3 blocks above + 3 below as plain text):
<<<NEIGHBORS>>>
{neighbors_text[:4000]}
<<<END NEIGHBORS>>>

DOC_META (optional JSON): {meta_str}

REWRITE REQUIREMENTS
- Explain what the {kind} shows and how to read it.
- Tie specific values/labels in the primary payload to statements.
- Keep units as printed (no conversion).
- Mention uncertainty/significance if present.
- If inputs conflict, state the conflict.
- No headings, no bullets—just one paragraph.
- If not enough information: output exactly 'Insufficient context'.
"""
    combined = f"{system}\n\n{user}"
    contents = _content_from_texts(combined)

    last = None
    for m in _prefs(CONF.text_prefs[0], CONF.text_prefs):
        try:
            resp = genai.GenerativeModel(m).generate_content(
                contents, generation_config=_GENCFG_TEXT, safety_settings=_SAFETY
            )
            text = (resp.text or "").strip()
            if text:
                CACHE.set("ctx_narr", cache_key, text)
                return text
        except Exception as e:
            last = e
            continue
    raise last or RuntimeError("Gemini contextualize failed")

# ----------------------------
# Evidence-bound Answering
# ----------------------------
@backoff.on_exception(backoff.expo, Exception, max_tries=5)
def answer(pack_text: str, q: str) -> str:
    sys = (
        "You are a careful scientific assistant. Answer ONLY using the provided PACK. "
        "If the PACK lacks sufficient information, respond exactly with 'Insufficient evidence'. "
        "When you cite, quote short supporting spans in double quotes. "
        "For numeric results, show a brief calculation if applicable."
    )

    user = f"""PACK:
<<<
{pack_text}
>>>

QUESTION:
{q}

FORMAT:
- If answerable: one short paragraph (<= 6 sentences) + 1-3 bullet quotes with minimal spans.
- If not answerable from PACK: output exactly 'Insufficient evidence'.
"""
    combined = f"{sys}\n\n{user}"
    contents = _content_from_texts(combined)

    last = None
    for m in _prefs(CONF.text_prefs[0], CONF.text_prefs):
        try:
            resp = genai.GenerativeModel(m).generate_content(
                contents, generation_config=_GENCFG_TEXT
            )
            text = (resp.text or "").strip()
            if text:
                return text
        except Exception as e:
            last = e
            continue
    raise last or RuntimeError("Gemini answer failed")
