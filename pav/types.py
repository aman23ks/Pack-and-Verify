from typing import List, Dict, Any, Tuple, TypedDict

class Bundle(TypedDict):
    id: str
    kind: str
    page: int
    text_main: str
    context_pre: List[str]
    context_post: str
    cost: int
