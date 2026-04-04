from __future__ import annotations

import re

from archivist.types import Anchor, Page


_CHAPTER_RE = re.compile(
    r"^\s*(?:[\"'“”‘’]+\s*)*(chapter)\s+([ivxlcdm]+|\d+)\b",
    re.IGNORECASE,
)


def roman_to_int(s: str) -> int:
    s = s.strip().upper()
    if not s:
        raise ValueError("Empty roman numeral")

    values = {"I": 1, "V": 5, "X": 10, "L": 50, "C": 100, "D": 500, "M": 1000}
    total = 0
    prev = 0
    for ch in reversed(s):
        if ch not in values:
            raise ValueError(f"Invalid roman numeral: {s}")
        v = values[ch]
        if v < prev:
            total -= v
        else:
            total += v
            prev = v
    return total


def extract_anchors(pages: list[Page]) -> list[Anchor]:
    """Find chapter-heading pages and return anchors sorted by chapter number."""

    anchors: list[Anchor] = []
    for p in pages:
        m = _CHAPTER_RE.match(p.text)
        if not m:
            continue
        chap_raw = m.group(2)
        try:
            chap_num = int(chap_raw) if chap_raw.isdigit() else roman_to_int(chap_raw)
        except ValueError:
            continue
        anchors.append(Anchor(chapter_num=chap_num, page_id=p.page_id))

    # De-duplicate by chapter number (keep first occurrence).
    seen: set[int] = set()
    deduped: list[Anchor] = []
    for a in sorted(anchors, key=lambda x: (x.chapter_num, x.page_id)):
        if a.chapter_num in seen:
            continue
        seen.add(a.chapter_num)
        deduped.append(a)

    return deduped
