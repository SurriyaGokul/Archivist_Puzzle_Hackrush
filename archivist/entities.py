from __future__ import annotations

import re
from collections import Counter

import numpy as np

from archivist.data import normalize_text
from archivist.types import Page

# Common English words that look like proper nouns but aren't.
_COMMON_WORDS = frozenset({
    "The", "A", "An", "In", "On", "It", "He", "She", "We", "They",
    "But", "And", "Or", "If", "So", "No", "Yes", "My", "His", "Her",
    "This", "That", "What", "When", "Where", "Who", "How", "Not",
    "Chapter", "Mr", "Mrs", "Miss", "Sir", "Dr", "Lord", "Lady",
    "I", "You", "Your", "Our", "Their", "There", "Here", "Then",
    "Now", "Well", "Oh", "Ah", "For", "With", "From", "At", "To",
    "Of", "As", "Is", "Was", "Be", "Do", "Did", "Has", "Had",
    "Could", "Would", "Should", "Will", "Shall", "May", "Can",
    "Very", "Much", "Some", "One", "Two", "Three", "Four", "Five",
    "Let", "By", "All", "Just", "Still", "About", "After", "Before",
    "Over", "Under", "Been", "Being", "These", "Those", "Such",
    "Than", "Its", "Have", "Them", "Into", "Only", "More", "Most",
    "Each", "Every", "Both", "Any", "Other", "New", "Old", "Good",
    "Great", "Little", "Long", "First", "Last", "Own", "Part",
    "Back", "Down", "Up", "Out", "Off", "Away", "Even", "Never",
    "Yet", "Already", "Also", "Too", "Quite", "Rather", "Perhaps",
    "Indeed", "However", "Though", "Although", "While", "Because",
    "Since", "Until", "Unless", "Nor", "Neither", "Either",
    "Nothing", "Something", "Everything", "Anything", "Nobody",
    "Somebody", "Everybody", "Anybody", "Enough", "Several",
    "Many", "Few", "Half", "Another", "Next", "Dear", "Poor",
    "English", "French", "Indian", "British", "American",
})

_WORD_RE = re.compile(r"[A-Z][a-z]+")


def auto_detect_characters(pages: list[Page], *, min_count: int = 5) -> list[str]:
    """Find recurring proper nouns (likely character names) across all pages."""

    counts: Counter[str] = Counter()
    for p in pages:
        text = normalize_text(p.text)
        for m in _WORD_RE.finditer(text):
            word = m.group()
            if word not in _COMMON_WORDS and len(word) > 2:
                counts[word] += 1

    return [name for name, c in counts.most_common() if c >= min_count]


def _extract_names_from_window(text: str, characters: list[str]) -> set[str]:
    """Extract character names present in text."""

    text_lower = text.lower()
    found = set()
    for ch in characters:
        if ch.lower() in text_lower:
            found.add(ch)
    return found


def character_flow_matrix(
    texts: list[str],
    characters: list[str],
    *,
    tail_words: int = 80,
    head_words: int = 80,
) -> np.ndarray:
    """NxN matrix: W[i,j] = Jaccard similarity of characters in tail(i) and head(j).

    Higher values mean pages i and j share characters at their boundary,
    suggesting they are adjacent.
    """

    n = len(texts)
    if n == 0 or not characters:
        return np.zeros((n, n), dtype=np.float32)

    # Extract tail/head character sets.
    tail_chars: list[set[str]] = []
    head_chars: list[set[str]] = []
    for t in texts:
        t = normalize_text(t)
        words = t.split()
        tail_text = " ".join(words[-tail_words:])
        head_text = " ".join(words[:head_words])
        tail_chars.append(_extract_names_from_window(tail_text, characters))
        head_chars.append(_extract_names_from_window(head_text, characters))

    w = np.zeros((n, n), dtype=np.float32)
    for i in range(n):
        if not tail_chars[i]:
            continue
        for j in range(n):
            if i == j or not head_chars[j]:
                continue
            inter = len(tail_chars[i] & head_chars[j])
            if inter == 0:
                continue
            union = len(tail_chars[i] | head_chars[j])
            w[i, j] = float(inter) / float(union)

    return w
