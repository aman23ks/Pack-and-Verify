from typing import List, Tuple
from collections import defaultdict

def pack(matches, budget: int, lam: float = 0.6, per_section_cap: int = 999) -> Tuple[list, int]:

    for m in matches:
        md = m.metadata
        md["cost"] = max(1, md.get("tokens_estimate", 200))
        md["gain"] = m.score
        md["sect"] = f'{md.get("kind","")}-p{md.get("page",-1)}'

    chosen, used, chosen_ids = [], 0, set()
    sect_counts = defaultdict(int)

    while True:
        best, best_score = None, -1e18
        for m in matches:
            if m.id in chosen_ids: continue
            c = m.metadata["cost"]
            if used + c > budget: continue
            if sect_counts[m.metadata["sect"]] >= per_section_cap: continue
            red = lam * max([abs(m.score - cm.score) for cm in chosen] or [0])
            score = (m.metadata["gain"] - red) / c
            if score > best_score:
                best, best_score = m, score
        if not best: break
        chosen.append(best); chosen_ids.add(best.id); used += best.metadata["cost"]
        sect_counts[best.metadata["sect"]] += 1
    return chosen, used

def render_pack(chosen) -> str:
    blocks=[]
    for m in chosen:
        md=m.metadata
        block=f"[{md['kind']} p{md['page']}]\n"
        if md["context_pre"]: block+="\n".join(md["context_pre"])+"\n"
        block+=md["text_main"]
        if md["context_post"]: block+="\n"+md["context_post"]
        blocks.append(block)
    return "\n\n---\n\n".join(blocks)
