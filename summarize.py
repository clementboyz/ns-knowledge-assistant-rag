import re
from typing import List, Dict, Any, Tuple


def _normalize_header(line: str) -> str:
    """
    Convert headings like '# Borrow', '## Borrow', 'Borrow' -> 'borrow'
    """
    line = line.strip()
    line = re.sub(r"^#+\s*", "", line)  # remove leading markdown #'s
    return line.strip().lower()


def _find_header_line(lines: List[str], header: str) -> int:
    target = header.strip().lower()
    for i, l in enumerate(lines):
        if _normalize_header(l) == target:
            return i
    return -1


def _is_section_header(line: str) -> bool:
    tag = _normalize_header(line)
    return tag in ("borrow", "return", "notes")


def _slice_section(lines: List[str], start_idx: int) -> List[str]:
    """
    Take lines after header until next section header or end.
    Stops on Borrow/Return/Notes headers (supports markdown like '## Borrow').
    """
    out = []
    for j in range(start_idx + 1, len(lines)):
        if _is_section_header(lines[j]):
            break
        out.append(lines[j])
    return out


def _extract_steps_from_lines(lines: List[str], max_steps: int = 6) -> List[str]:
    step_like = []
    for l in lines:
        s = l.strip()
        if not s:
            continue

        # Match "1. xxx" / "1) xxx" / "- xxx" / "* xxx"
        if re.match(r"^(\d+[\).\:]|\-|\*)\s+", s):
            cleaned = re.sub(r"^(\d+[\).\:]|\-|\*)\s+", "", s).strip()
            if cleaned:
                step_like.append(cleaned)

    if step_like:
        return step_like[:max_steps]

    # fallback: sentence split (only within the section slice)
    text = " ".join([x.strip() for x in lines if x.strip()])
    if not text:
        return []
    sents = re.split(r"(?<=[.!?])\s+", text)
    sents = [s.strip() for s in sents if s.strip()]
    return sents[:max_steps]


def build_final_answer(results: List[Dict[str, Any]]) -> Tuple[List[str], List[str]]:
    """
    Returns (borrow_steps, return_steps) extracted strictly from the top evidence chunk.
    Supports markdown headers like '## Borrow' and '## Return'.
    """
    if not results:
        return [], []

    text = results[0]["text"]
    lines = text.splitlines()

    borrow_idx = _find_header_line(lines, "Borrow")
    return_idx = _find_header_line(lines, "Return")

    borrow_steps: List[str] = []
    return_steps: List[str] = []

    if borrow_idx != -1:
        borrow_lines = _slice_section(lines, borrow_idx)
        borrow_steps = _extract_steps_from_lines(borrow_lines, max_steps=6)

    if return_idx != -1:
        return_lines = _slice_section(lines, return_idx)
        return_steps = _extract_steps_from_lines(return_lines, max_steps=6)

    # If neither section found, fallback to generic extraction
    if borrow_idx == -1 and return_idx == -1:
        borrow_steps = _extract_steps_from_lines(lines, max_steps=6)

    return borrow_steps, return_steps
