import logging
import os
from typing import Optional, Dict, Any, List
from dotenv import load_dotenv
from openai import OpenAI

logger = logging.getLogger(__name__)

# -------------------------------------------------------------------
# OpenAI setup
# -------------------------------------------------------------------
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    logger.warning("OPENAI_API_KEY is not set – OpenAI calls will fail.")

client = OpenAI(api_key=OPENAI_API_KEY)

TEXT_MODEL = os.getenv("OPENAI_TEXT_MODEL", "gpt-4.1-mini")


def _chat(prompt: str, max_tokens: int = 2048, temperature: float = 0.2) -> str:
    """
    Simple wrapper around OpenAI chat completions.
    """
    if not OPENAI_API_KEY:
        logger.error("_chat() called without OPENAI_API_KEY.")
        return ""

    try:
        resp = client.chat.completions.create(
            model=TEXT_MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=temperature,
        )
        text = resp.choices[0].message.content or ""
        return text.strip()
    except Exception as e:
        logger.error("OpenAI chat completion error: %s", e)
        return ""

def vision(image_base64: str, image_mime_type: str, prompt: Optional[str] = None) -> str:
    """
    TEMP STUB: We are not actually sending the image to OpenAI here.

    For now, we return an empty string, so contextualize() relies only on
    HTML + surrounding text. This avoids Gemini entirely and is still
    useful for retrieval.

    If you later want true vision, we can wire OpenAI's image input here.
    """
    logger.info("vision() stub called – returning empty summary (no Gemini).")
    return ""

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
    vision_summary: optional structured description from vision() – currently often ""
    doc_meta: {"page": int, "element_id": str} – used only in prompt text.
    """
    kind = kind.lower()
    if kind not in {"image", "table"}:
        raise ValueError(f"Unsupported kind for contextualize(): {kind}")

    if not OPENAI_API_KEY:
        logger.error("contextualize() called without OPENAI_API_KEY.")
        return ""

    page_str = f"Page: {doc_meta.get('page')}" if doc_meta and "page" in doc_meta else ""
    elem_str = f"element_id: {doc_meta.get('element_id')}" if doc_meta and "element_id" in doc_meta else ""
    header_line = f"Document context ({page_str} {elem_str})".strip()

    item_label = "FIGURE" if kind == "image" else "TABLE"

    prompt = f"""
You are helping build a high-quality retrieval index over scientific PDFs.
Your goal is to produce a *self-contained, detailed narrative* for each figure or table
using ONLY the information explicitly available in the provided context.

Treat this as a {item_label}. You DO NOT see the raw pixels; rely strictly on:
- the HTML element (<img> or <table>),
- the nearby text immediately above and below,
- and an optional machine-generated visual summary (if available).

Use ALL of the following, in this order:

TEXT_ABOVE:
{text_above or "[none]"}

ITEM_CONTENT_HTML:
{text_html or "[none]"}

VISION_SUMMARY:
{vision_summary or "[none]"}

TEXT_BELOW:
{text_below or "[none]"}

TASK:
- Write a single, cohesive, and fully self-contained narrative that:
  1) Describes what the {item_label} likely shows, based only on the above evidence.
  2) Explains how it fits into the surrounding discussion (TEXT_ABOVE/TEXT_BELOW).
  3) For tables: mention headers, columns, and interpret visible numeric values if available.
  4) For figures/charts: mention axes, trends, comparisons, or relationships described in text.
  5) If the HTML or vision summary seems incomplete, clearly note the ambiguity—do not speculate.

CONSTRAINTS:
- Base your narrative ONLY on the given context.
- Do NOT infer information from external knowledge or the internet.
- Do NOT invent numbers, labels, or results not supported by the evidence.
- Return plain text ONLY (no lists, markdown, or JSON).

{header_line}
""".strip()

    return _chat(prompt, max_tokens=4096, temperature=0.2)


# -------------------------------------------------------------------
# Embeddings wrapper (used by discourse_units._embed)
# -------------------------------------------------------------------

def embed(texts: List[str]) -> List[List[float]]:
    """
    Kept for backwards-compatibility with discourse_units._embed.

    Internally delegates to pav.clients.embeddings.embed(), which now
    uses OpenAI embeddings.
    """
    from pav.clients.embeddings import embed as _emb
    return _emb(texts)


# -------------------------------------------------------------------
# Generic QA answer() used by pav/qa/answer.py
# -------------------------------------------------------------------

def answer(*args, **kwargs) -> str:
    """
    QA wrapper over retrieved context, now using OpenAI chat completions.

    Supported patterns:
      - answer(prompt: str)
      - answer(question: str, context: str)
      - answer(question=..., context=...)
    """
    if not OPENAI_API_KEY:
        logger.error("answer() called without OPENAI_API_KEY.")
        return ""

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

    return _chat(prompt, max_tokens=1024, temperature=0.2)
