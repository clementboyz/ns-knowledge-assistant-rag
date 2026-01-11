import re
from typing import List, Dict, Any, Tuple


def _extract_section(text: str, header: str) -> str:
    """
    Extract content under a markdown header like 'Borrow' or 'Return'.
    Works for simple docs like:
    Borrow
    1. ...
    2. ...
    Return
    1. ...
    """
    lines = text.splitlines()
    header_idx = None
    for i, l in enumerate(lines):
        if l.strip().lower() == header.lower():
            header_idx = i
            break
    if header_idx is None:
        return ""

    # take until next major header
    out = []
    for j in range(header_idx + 1, len(lines)):
        if lines[j].strip().lower() in ("borrow", "return", "notes"):
            break
        out.append(lines[j])
    return "\n".join(out).strip()


def extract_steps(text: str, max_steps: int = 6) -> List[str]:
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    step_like = []
    for l in lines:
        if re.match(r"^(\d+[\).\:]|\-|\*)\s+", l):
            # remove leading numbering/bullets for cleaner UI
            cleaned = re.sub(r"^(\d+[\).\:]|\-|\*)\s+", "", l).strip()
            if cleaned:
                step_like.append(cleaned)

    if step_like:
        return step_like[:max_steps]

    # fallback: short sentence split
    sents = re.split(r"(?<=[.!?])\s+", " ".join(lines))
    sents = [s.strip() for s in sents if s.strip()]
    return sents[:max_steps]


def build_final_answer(results: List[Dict[str, Any]]) -> Tuple[List[str], List[str]]:
    """
    Returns (borrow_steps, return_steps)
    Extract from top evidence chunk.
    """
    if not results:
        return [], []

    text = results[0]["text"]

    borrow_text = _extract_section(text, "Borrow") or text
    return_text = _extract_section(text, "Return")

    borrow_steps = extract_steps(borrow_text, max_steps=6)
    return_steps = extract_steps(return_text, max_steps=6) if return_text else []

    return borrow_steps, return_steps
