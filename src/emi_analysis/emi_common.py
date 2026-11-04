"""Definitions shared by every contextual-EMI stage: eras, lexicons, the paper's text rules, and hashing.

See docs/superpowers/specs/2026-11-04-contextual-emi-design.md.
"""

from __future__ import annotations

import re

import numpy as np

# Eight eras of nine Congresses (1879-2022, 18 years each).
ERAS: list[tuple[int, int]] = [(46 + 9 * i, 54 + 9 * i) for i in range(8)]
N_ERAS = len(ERAS)

_MASK = (1 << 64) - 1


def era_of(congress) -> np.ndarray:
    """Era index (0-7) for each Congress number; -1 outside 46-117."""
    congress = np.asarray(congress, dtype=np.int64)
    era = (congress - ERAS[0][0]) // 9
    return np.where((congress >= ERAS[0][0]) & (congress <= ERAS[-1][1]), era, -1)


def era_label(era: int) -> str:
    lo, hi = ERAS[era]
    return f"{lo}-{hi}"


def load_lexicons(config) -> tuple[list[str], list[str]]:
    """The evidence and intuition keywords (lower-cased, de-duplicated, in file order)."""
    from .train import load_lexicon

    evidence = list(dict.fromkeys(w.lower() for w in load_lexicon(config.evidence_lexicon_path)))
    intuition = list(dict.fromkeys(w.lower() for w in load_lexicon(config.intuition_lexicon_path)))
    both = set(evidence) & set(intuition)
    if both:
        raise ValueError(f"Keywords in both lexicons: {sorted(both)}")
    return evidence, intuition


# ------------------------------------------------------------------ the paper's rules (compute_sbert_avg_lexicon.py)

_PAPER_PATTERNS = [re.compile(p) for p in (r"\.\.+", r"\-\-+", r"__+", r"\*\*+", r"\s+")]


def paper_preprocess(text: str) -> str:
    for pattern in _PAPER_PATTERNS:
        text = pattern.sub(" ", text)
    return text


def top100_words() -> set[str]:
    from wordfreq import top_n_list

    return set(top_n_list("en", 100))


def word_stats(text: str, top100: set[str]) -> tuple[int, float]:
    """Word count and the fraction of words among the 100 most common English words, as the paper computes them.

    The paper splits on whitespace after its punctuation clean-up and compares case-sensitively.
    """
    words = paper_preprocess(text).split()
    if not words:
        return 0, 0.0
    return len(words), sum(w in top100 for w in words) / len(words)


def n_chunks_150(word_count) -> np.ndarray:
    """Number of chunks the paper's chunker makes: 150-word chunks, a last chunk under 50 words merged."""
    wc = np.asarray(word_count, dtype=np.int64)
    full, rest = np.divmod(wc, 150)
    chunks = full + (rest > 0)
    merge = (chunks > 1) & (rest > 0) & (rest < 50)
    return np.maximum(1, chunks - merge)


def scorable(word_count, top100_frac) -> np.ndarray:
    """The paper's filters: more than 10 words and at least 5% of words among the top 100."""
    return (np.asarray(word_count) > 10) & (np.asarray(top100_frac) >= 0.05)


# ------------------------------------------------------------------ deterministic hashing

def mix64(x: np.ndarray) -> np.ndarray:
    """splitmix64 finalizer on uint64 arrays (wrapping arithmetic)."""
    x = np.asarray(x, dtype=np.uint64)
    with np.errstate(over="ignore"):
        x = x + np.uint64(0x9E3779B97F4A7C15)
        x = (x ^ (x >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
        x = (x ^ (x >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
    return x ^ (x >> np.uint64(31))


def hash_key(seed: int, *parts) -> np.ndarray:
    """A uniform 64-bit key for each combination of integer parts (arrays broadcast)."""
    key = mix64(np.uint64(seed & _MASK))
    for part in parts:
        key = mix64(key ^ np.asarray(part, dtype=np.int64).astype(np.uint64))
    return key


def mix64_int(x: int) -> int:
    x = (x + 0x9E3779B97F4A7C15) & _MASK
    x = ((x ^ (x >> 30)) * 0xBF58476D1CE4E5B9) & _MASK
    x = ((x ^ (x >> 27)) * 0x94D049BB133111EB) & _MASK
    return x ^ (x >> 31)


def hash_key_int(seed: int, *parts: int) -> int:
    key = mix64_int(seed & _MASK)
    for part in parts:
        key = mix64_int(key ^ (part & _MASK))
    return key
