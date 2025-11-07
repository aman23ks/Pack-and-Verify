from ..clients.gemini import answer as g_answer

def answer(pack_text: str, question: str) -> str:
    return g_answer(pack_text, question)
