# pav/clients/gemini.py
import base64
import logging
import os
from typing import Optional, Dict, List, Any

import google.generativeai as genai
from google.generativeai.types import GenerationConfig

logger = logging.getLogger(__name__)

# -------------------------------------------------------------------
# Gemini setup
# -------------------------------------------------------------------

GEMINI_API_KEY = "AIzaSyDw0UjO1kZVQ5j1oplJhusJ8IbisPkRfD8"
if not GEMINI_API_KEY:
    logger.warning("GEMINI_API_KEY is not set – Gemini calls will fail.")
else:
    genai.configure(api_key=GEMINI_API_KEY)

# Models to try, in order
VISION_MODELS = [
    os.getenv("GEMINI_VISION_MODEL", "gemini-2.5-flash"),
]

TEXT_MODELS = [
    os.getenv("GEMINI_TEXT_MODEL_PRIMARY", "gemini-2.5-pro"),
    os.getenv("GEMINI_TEXT_MODEL_FALLBACK", "gemini-2.5-flash"),
]

# Embedding model
EMBED_MODEL = os.getenv("GEMINI_EMBED_MODEL", "models/text-embedding-004")

# -------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------


def _safe_text(resp) -> str:
    """
    Convert a Gemini response to plain text safely.
    """
    # Newer SDK: resp.text exists
    try:
        if hasattr(resp, "text") and isinstance(resp.text, str):
            return resp.text
    except Exception:
        pass

    # Fallback: walk candidates / parts
    try:
        parts: List[str] = []
        for cand in getattr(resp, "candidates", []) or []:
            content = getattr(cand, "content", None)
            if not content:
                continue
            for part in getattr(content, "parts", []) or []:
                txt = getattr(part, "text", None)
                if isinstance(txt, str):
                    parts.append(txt)
        return "\n".join(parts).strip()
    except Exception:
        return ""


# -------------------------------------------------------------------
# Vision: parse image into structured text
# -------------------------------------------------------------------


def vision(image_base64: str, image_mime_type: str, prompt: Optional[str] = None) -> str:
    """
    Call a vision-capable Gemini model with an image and an optional text prompt.

    Returns a *text* description (can be JSON or free-form) that we then feed into
    contextualize().
    """
    if not GEMINI_API_KEY:
        logger.error("Gemini vision() called without GEMINI_API_KEY.")
        return ""

    if prompt is None:
        prompt = (
            "You are a vision parser for scientific PDFs.\n\n"
            "TASK:\n"
            "- Look at the image and describe its *structure* in detail.\n"
            "- If it is a chart or table-like figure, extract:\n"
            "  * chart_type (e.g., line_plot, bar_chart, scatter_plot, illustration, etc.)\n"
            "  * axis labels, units, tick labels\n"
            "  * legend entries and what they represent\n"
            "  * for each series, the data points (with axis values)\n"
            "- If the image is not a chart (e.g., an architecture diagram, qualitative example),\n"
            "  describe the key components and their relationships.\n\n"
            "Format:\n"
            "- Prefer machine-readable JSON when you can.\n"
            "- If JSON is not natural, provide a very detailed plain-text description.\n"
        )

    try:
        image_bytes = base64.b64decode(image_base64)
    except Exception as e:
        logger.error("vision(): failed to decode base64 image: %s", e)
        return ""

    contents = [
        {
            "role": "user",
            "parts": [
                {"text": prompt},
                {
                    "inline_data": {
                        "mime_type": image_mime_type or "image/png",
                        "data": image_bytes,
                    }
                },
            ],
        }
    ]

    for model_name in VISION_MODELS:
        try:
            model = genai.GenerativeModel(model_name)
            resp = model.generate_content(
                contents,
                generation_config=GenerationConfig(
                    temperature=0.0,
                    top_p=0.9,
                    max_output_tokens=2048,
                ),
            )
            text = _safe_text(resp)
            if text:
                return text.strip()
        except Exception as e:
            logger.warning("Gemini vision error for model %s: %s", model_name, e)

    logger.error("Gemini vision() failed for all models.")
    return ""


# -------------------------------------------------------------------
# Contextualize: turn HTML + context (and optional vision summary)
# into a long, detailed narrative for retrieval
# -------------------------------------------------------------------


def contextualize(
    *,
    kind: str,
    text_html: str,
    text_above: str,
    text_below: str,
    vision_summary: Optional[str] = None,
    doc_meta: Optional[Dict[str, Any]] = None,
) -> str:
    """
    Build a very detailed narrative in-context for an image or table.

    kind: "image" | "table"
    text_html: HTML for the image or table (e.g., <img ...> or <table>...</table>)
    text_above: surrounding text *before* the element (already concatenated)
    text_below: surrounding text *after* the element (already concatenated)
    vision_summary: optional structured description from vision()
    doc_meta: {"page": int, "element_id": str} or similar – only used in prompt text.
    """
    if not GEMINI_API_KEY:
        logger.error("Gemini contextualize() called without GEMINI_API_KEY.")
        return ""

    kind = kind.lower()
    if kind not in {"image", "table"}:
        raise ValueError(f"Unsupported kind for contextualize(): {kind}")

    page_str = ""
    if doc_meta and "page" in doc_meta:
        page_str = f"Page: {doc_meta['page']}"
    elem_str = ""
    if doc_meta and "element_id" in doc_meta:
        elem_str = f"element_id: {doc_meta['element_id']}"

    header_line = f"Document context ({page_str} {elem_str})".strip()

    role_line = (
        "You are helping build a high-quality retrieval index over scientific PDFs.\n"
        "Your job is to transform each figure or table, together with its local context,\n"
        "into a *self-contained, very detailed narrative* that explains exactly what it shows\n"
        "and how it relates to the surrounding text."
    )

    if kind == "image":
        item_label = "FIGURE"
        extra_instruction = (
            "- Treat this as an IMAGE (figure). It may be a chart, plot, qualitative examples,\n"
            "  architecture diagram, or other visual.\n"
        )
    else:
        item_label = "TABLE"
        extra_instruction = (
            "- Treat this as a TABLE. Assume the HTML and context are authoritative for column\n"
            "  names, row labels, and values, even if some formatting is imperfect.\n"
        )

    detailed_instructions = f"""
{role_line}

{extra_instruction}

You are given the following pieces of information in this exact order:

1. TEXT_ABOVE:
   This is the text immediately before the {item_label} in the PDF. It provides local narrative
   context, definitions, and explanations leading into the {item_label}.

2. ITEM_CONTENT_HTML:
   This is the HTML representation of the {item_label}. It may contain:
   - The rendered HTML (<img> tag or <table> markup)
   - Any caption or inline description that the parser associated with it.

3. VISION_SUMMARY:
   A machine-generated description of the figure/table structure (axes, series, numeric values,
   legend entries, etc.). Use this to recover quantitative details that may be hard to read from
   the HTML alone. If it seems inconsistent with the TEXT_ABOVE/TEXT_BELOW, prioritize the
   human-written text and clearly mention the inconsistency.

4. TEXT_BELOW:
   This is the text immediately after the {item_label} in the PDF. It often contains follow-up
   explanation, discussion of trends, or references to specific rows/columns/curves.

TASK:

- Using ALL of the above, write a *single, long, self-contained narrative* that:
  1) Explicitly describes what the {item_label} shows in as much detail as possible.
  2) Explains how it fits into the surrounding discussion (TEXT_ABOVE/TEXT_BELOW).
  3) For charts:
     - Name the axes and units.
     - List the legend entries and what each series represents.
     - Describe how the curves/bars differ numerically (e.g., “Ferret-v2 achieves 81.0 vs 76.7 on VQAv2”).
     - Summarize the main trends and conclusions.
  4) For tables:
     - List all columns, row groups, and key cells.
     - Explain what each numeric value represents conceptually (e.g., “Recall@1 on RefCOCOg”).
     - Highlight any important comparisons (best rows, deltas, ablations, etc.).
  5) Explicitly tie each important conclusion to evidence from the table/figure.

- The output should be detailed enough that a retrieval model could answer questions about this
  figure/table *without* needing to see the original PDF again.

CONSTRAINTS:

- Do NOT invent numbers or tasks that are not clearly supported.
- If something is ambiguous, say so and explain what is and is not clear.
- Do NOT respond in JSON. Return a single, human-readable narrative paragraph (or a few
  paragraphs) in plain text.

-----

{header_line}

TEXT_ABOVE:
{text_above or "[none]"}

ITEM_CONTENT_HTML:
{text_html or "[none]"}

VISION_SUMMARY:
{vision_summary or "[none]"}

TEXT_BELOW:
{text_below or "[none]"}
""".strip()

    for model_name in TEXT_MODELS:
        try:
            model = genai.GenerativeModel(model_name)
            resp = model.generate_content(
                [detailed_instructions],
                generation_config=GenerationConfig(
                    temperature=0.2,
                    top_p=0.9,
                    max_output_tokens=4096,
                ),
            )
            text = _safe_text(resp)
            if text:
                return text.strip()
        except Exception as e:
            logger.warning("Gemini contextualize error for model %s: %s", model_name, e)

    logger.error("Gemini contextualize() failed for all models; returning raw context.")
    # As last resort, just concatenate everything so something goes in the DB
    return detailed_instructions


# -------------------------------------------------------------------
# Embeddings for Pinecone
# -------------------------------------------------------------------


def embed(texts: List[str]) -> List[List[float]]:
    """
    Return embeddings for a list of texts using Gemini.

    This is what Pinecone expects:
      - `texts` is a list of strings
      - return value is a list of float vectors (one per text)
    """
    if isinstance(texts, str):
        texts = [texts]

    if not GEMINI_API_KEY:
        raise RuntimeError("GEMINI_API_KEY is not set; cannot call Gemini embed().")

    if not texts:
        return []

    out: List[List[float]] = []

    for t in texts:
        if not t:
            out.append([])
            continue

        try:
            resp = genai.embed_content(
                model=EMBED_MODEL,
                content=t,
                task_type="RETRIEVAL_DOCUMENT",
            )
        except Exception as e:
            logger.error("Gemini embed_content() error: %s", e)
            out.append([])
            continue

        # Handle various possible response shapes
        emb = None
        try:
            # dict-like
            emb = resp["embedding"]["values"]
        except Exception:
            try:
                emb_obj = getattr(resp, "embedding", None)
                if emb_obj is not None and hasattr(emb_obj, "values"):
                    emb = emb_obj.values
                else:
                    emb_list = getattr(resp, "embeddings", None)
                    if emb_list and hasattr(emb_list[0], "values"):
                        emb = emb_list[0].values
            except Exception as e2:
                logger.error("Unexpected embed_content() response shape: %r", e2)
                emb = None

        if emb is None:
            out.append([])
        else:
            out.append(list(emb))

    return out


# -------------------------------------------------------------------
# Generic QA answer() used by pav/qa/answer.py
# -------------------------------------------------------------------


def answer(*args, **kwargs) -> str:
    """
    Very defensive wrapper so pav.qa.answer can call this in multiple ways.

    Supported patterns:
      - answer(prompt: str)
      - answer(question: str, context: str)
      - answer(question=..., context=...)
    """
    if not GEMINI_API_KEY:
        logger.error("Gemini answer() called without GEMINI_API_KEY.")
        return ""

    # Extract question / context from positional or keyword args
    question = kwargs.pop("question", None)
    context = kwargs.pop("context", None)

    # Positional fallback
    if question is None and len(args) >= 1:
        question = args[0]
    if context is None and len(args) >= 2:
        context = args[1]

    # Simple “just a prompt” mode
    if context is None and len(args) == 1 and not kwargs:
        prompt = str(question) if question is not None else ""
    else:
        # QA-over-context mode
        q_text = str(question) if question is not None else ""
        c_text = str(context) if context is not None else ""
        prompt = (
            "You are a QA assistant answering questions over retrieved chunks from a document.\n\n"
            "CONTEXT:\n"
            f"{c_text}\n\n"
            "QUESTION:\n"
            f"{q_text}\n\n"
            "TASK:\n"
            "- Answer the question using only the information in CONTEXT.\n"
            "- Be concise but clear.\n"
            "- If the context is insufficient or the answer is not supported, say that explicitly.\n"
        )

    last_error = None
    for model_name in TEXT_MODELS:
        try:
            model = genai.GenerativeModel(model_name)
            resp = model.generate_content(
                [prompt],
                generation_config=GenerationConfig(
                    temperature=0.2,
                    top_p=0.9,
                    max_output_tokens=1024,
                ),
            )
            text = _safe_text(resp)
            if text:
                return text.strip()
        except Exception as e:
            last_error = e
            logger.warning("Gemini answer() error for model %s: %s", model_name, e)

    logger.error("Gemini answer() failed for all models. Last error: %s", last_error)
    return ""
