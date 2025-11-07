import base64, backoff, google.generativeai as genai
from typing import List
from ..config import CONF
from ..cache import CACHE

genai.configure(api_key=CONF.google_key)

def _prefs(primary: str, fallbacks: tuple) -> List[str]:
    return [primary] + [m for m in fallbacks if m != primary]

@backoff.on_exception(backoff.expo, Exception, max_tries=5)
def embed(texts: List[str]) -> List[List[float]]:
    out = []
    for t in texts:
        c = CACHE.get("embed", t)
        if c: out.append(c); continue
        emb = genai.embed_content(model=CONF.embed_model, content=t)["embedding"]
        CACHE.set("embed", t, emb)
        out.append(emb)
    return out

@backoff.on_exception(backoff.expo, Exception, max_tries=5)
def vision(image_bytes: bytes, context: str) -> str:
    prompt = (
        "Extract faithful text/labels/numbers/units from the image. "
        "Use nearby context ONLY to disambiguate; do not invent.\n\n"
        f"Nearby context:\n{context[:2000]}"
    )
    parts = [{"text": prompt},
             {"inline_data": {"mime_type": "image/png",
                              "data": base64.b64encode(image_bytes).decode()}}]
    last = None
    for m in _prefs(CONF.vision_prefs[0], CONF.vision_prefs):
        try:
            key = f"vision:{m}:{hash(image_bytes)}:{hash(context[:512])}"
            c = CACHE.get("vision", key)
            if c: return c
            resp = genai.GenerativeModel(m).generate_content(
                parts, generation_config={"temperature": 0.0},
                safety_settings={"HARASSMENT":"block_none","HATE_SPEECH":"block_none",
                                 "SEXUAL":"block_none","DANGEROUS":"block_none"})
            text = (resp.text or "").strip()
            if text:
                CACHE.set("vision", key, text)
                return text
        except Exception as e:
            last = e; continue
    raise last or RuntimeError("Gemini vision failed")

@backoff.on_exception(backoff.expo, Exception, max_tries=5)
def answer(pack_text: str, q: str) -> str:
    sys = ("Answer ONLY from PACK. Quote short spans or say 'Insufficient evidence'. "
           "If numeric, show a brief calc.")
    prompt = [{"text": sys + "\n\nPACK:\n" + pack_text + "\n\nQuestion: " + q}]
    last = None
    for m in _prefs(CONF.text_prefs[0], CONF.text_prefs):
        try:
            resp = genai.GenerativeModel(m).generate_content(
                prompt, generation_config={"temperature": 0.0})
            text = (resp.text or "").strip()
            if text: return text
        except Exception as e:
            last = e; continue
    raise last or RuntimeError("Gemini answer failed")
