# pav/ingest/build_bundles.py
from typing import List, Dict, Any
from ..types import Bundle
from ..config import CONF
from .discourse_units import build_ccus

def build(elements: List[Dict[str,Any]], doc_id: str) -> List[Bundle]:
    """
    CCU mode: build context-complete units from section structure.
    """
    if not CONF.CCU_ENABLE:
        raise RuntimeError("CCU mode is disabled. Set CCU_ENABLE=true in .env or config.py")
    return build_ccus(elements, doc_id)
