# pav/config.py
import os
from dataclasses import dataclass
from dotenv import load_dotenv

load_dotenv()

@dataclass
class Settings:
    # --- APIs ---
    unstructured_url: str = "https://api.unstructuredapp.io/general/v0/general"
    unstructured_key: str = os.getenv("UNSTRUCTURED_API_KEY", "")

    google_key: str = os.getenv("GOOGLE_API_KEY", "")

    pinecone_key: str = os.getenv("PINECONE_API_KEY", "")
    pine_region: str = os.getenv("PINECONE_REGION", "us-east-1")
    pine_cloud:  str = os.getenv("PINECONE_CLOUD", "aws")
    pine_index:  str = os.getenv("PINECONE_INDEX", "pav-quick")

    # --- Models ---
    embed_model: str      = os.getenv("EMBED_MODEL", "gemini-embedding-001")  # 768-d
    vision_prefs: tuple   = (
        os.getenv("VISION_PRIMARY", "gemini-2.5-pro"),
        os.getenv("VISION_FALLBACK", "gemini-2.5-flash"),
    )
    text_prefs: tuple     = (
        os.getenv("TEXT_PRIMARY", "gemini-2.5-pro"),
        os.getenv("TEXT_FALLBACK", "gemini-2.5-flash"),
    )

    # --- Pipeline knobs ---
    embed_batch:  int = int(os.getenv("EMBED_BATCH", "64"))
    upsert_batch: int = int(os.getenv("UPSERT_BATCH", "100"))
    token_budget: int = int(os.getenv("TOKEN_BUDGET", "3000"))

    # --- CCU (Context-Complete Units) ---
    # Number of neighboring text blocks above/below each table/image.
    # Set to 10/10 so Gemini sees a wider local context around each figure/table.
    CCU_STRUCT_NEIGHBORS: tuple = (10, 10)
    CCU_MAX_TOKENS:       int   = int(os.getenv("CCU_MAX_TOKENS", "3000"))
    CCU_OVERLAP_TOKENS:   int   = int(os.getenv("CCU_OVERLAP_TOKENS", "200"))
    CCU_ENABLE:           bool  = os.getenv("CCU_ENABLE", "true").lower() in ("1","true","yes","on")

CONF = Settings()

# Optional module-level aliases (in case any legacy import remains)
CCU_STRUCT_NEIGHBORS = CONF.CCU_STRUCT_NEIGHBORS
CCU_MAX_TOKENS       = CONF.CCU_MAX_TOKENS
CCU_OVERLAP_TOKENS   = CONF.CCU_OVERLAP_TOKENS
CCU_ENABLE           = CONF.CCU_ENABLE
