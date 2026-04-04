from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Page:
    page_id: int
    text: str


@dataclass(frozen=True)
class Anchor:
    chapter_num: int
    page_id: int
