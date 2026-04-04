from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

from archivist.types import Page


_UNICODE_TRANSLATION = str.maketrans(
    {
        "“": '"',
        "”": '"',
        "„": '"',
        "’": "'",
        "‘": "'",
        "—": "-",
        "–": "-",
        "…": "...",
        "\u00a0": " ",
    }
)


def normalize_text(text: str) -> str:
    """Normalize text for cheap lexical matching and boundary heuristics.

    Keeps content but reduces formatting artifacts:
    - join hyphen line-breaks
    - collapse whitespace/newlines
    - normalize common unicode punctuation
    - remove underscore-based emphasis markers
    """

    if text is None:
        return ""

    t = str(text)
    t = t.translate(_UNICODE_TRANSLATION)
    t = t.replace("\r\n", "\n").replace("\r", "\n")

    # Join common hyphenation across line breaks: "con-\ntext" -> "context".
    t = re.sub(r"(\w)-\n(\w)", r"\1\2", t)

    # Replace newlines/tabs with spaces.
    t = re.sub(r"\s*\n\s*", " ", t)
    t = t.replace("\t", " ")

    # Remove decorative underscores commonly used for italics in these books.
    t = t.replace("_", " ")

    # Collapse whitespace.
    t = re.sub(r"\s+", " ", t).strip()
    return t


def load_pages_csv(path: str | Path) -> list[Page]:
    """Load a CSV with columns: page,text.

    Uses pandas with settings that handle quoted multi-line fields.
    """

    path = Path(path)
    df = pd.read_csv(
        path,
        dtype={"page": "int64", "text": "string"},
        keep_default_na=False,
        na_filter=False,
        engine="python",
    )
    if "page" not in df.columns or "text" not in df.columns:
        raise ValueError(f"Expected columns ['page','text'] in {path}, got {list(df.columns)}")

    pages: list[Page] = []
    for row in df.itertuples(index=False):
        pages.append(Page(page_id=int(row.page), text=str(row.text)))

    _validate_pages(pages, source=str(path))
    return pages


def _validate_pages(pages: list[Page], *, source: str) -> None:
    if not pages:
        raise ValueError(f"No pages loaded from {source}")

    ids = [p.page_id for p in pages]
    if len(set(ids)) != len(ids):
        dupes = sorted({x for x in ids if ids.count(x) > 1})
        raise ValueError(f"Duplicate page ids in {source}: {dupes[:10]}")

    empty = [p.page_id for p in pages if not str(p.text).strip()]
    if empty:
        raise ValueError(f"Empty text for page ids in {source}: {empty[:10]}")


def word_window(text: str, *, head_words: int, tail_words: int) -> tuple[str, str]:
    """Return (head, tail) word windows from normalized text."""

    t = normalize_text(text)
    words = t.split()
    if not words:
        return "", ""

    head_words = max(1, int(head_words))
    tail_words = max(1, int(tail_words))

    head = " ".join(words[:head_words])
    tail = " ".join(words[-tail_words:])
    return head, tail
