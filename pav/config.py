import os
from dataclasses import dataclass
from dotenv import load_dotenv
load_dotenv()

@dataclass
class Settings:
    unstructured_url: str = "https://api.unstructuredapp.io/general/v0/general"
    unstructured_key: str = os.getenv("UNSTRUCTURED_API_KEY", "")
    google_key: str       = os.getenv("GOOGLE_API_KEY", "")
    pinecone_key: str     = os.getenv("PINECONE_API_KEY", "")
    pine_region: str      = os.getenv("PINECONE_REGION", "us-east-1")
    pine_cloud: str       = os.getenv("PINECONE_CLOUD", "aws")
    pine_index: str       = os.getenv("PINECONE_INDEX", "pav-quick")
    embed_model: str      = "gemini-embedding-001"      # 768-d
    vision_prefs: tuple   = ("gemini-2.5-pro", "gemini-2.5-flash")
    text_prefs: tuple     = ("gemini-2.5-pro", "gemini-2.5-flash")
    embed_batch: int      = 64
    upsert_batch: int     = 100
    token_budget: int     = 3000

CONF = Settings()
