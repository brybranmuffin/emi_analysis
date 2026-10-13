from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

from .config import BERT_MAX_LENGTH, GPT2_MAX_LENGTH, MIN_CHUNK_TOKENS

DEFAULT_HOLDOUT_SPEECH_COUNT = 20_000
DEFAULT_GENERAL_HOLDOUT_COUNT = 2_000


@dataclass
class SpeechRecord:
    speech_id: str
    congress: int
    text: str


@dataclass
class ChunkRecord:
    speech_id: str
    congress: int
    text: str
    chunk_index: int


def normalize_text(text: str) -> str:
    return " ".join((text or "").strip().split())


def hash_text(text: str) -> str:
    return hashlib.sha1(normalize_text(text).encode("utf-8")).hexdigest()


_PUNCTUATION = re.compile(r"[^\w\s]")


def dedup_key(text: str) -> str:
    """Hash of the text after lowercasing, stripping punctuation and collapsing whitespace."""
    simplified = _PUNCTUATION.sub("", (text or "").lower())
    return hashlib.sha1(" ".join(simplified.split()).encode("utf-8")).hexdigest()


def congress_from_year(year: int) -> int:
    """Congress number for a calendar year, using the EMI repo's convention (1879 -> 46th)."""
    return int((int(year) - 1789) // 2 + 1)


def deduplicate_speeches(speeches: Iterable[SpeechRecord]) -> tuple[list[SpeechRecord], dict[str, int]]:
    kept: list[SpeechRecord] = []
    seen: set[str] = set()
    counts_by_congress: dict[str, int] = {}

    for speech in speeches:
        key = hash_text(speech.text)
        if key in seen:
            counts_by_congress[str(speech.congress)] = counts_by_congress.get(str(speech.congress), 0) + 1
            continue
        seen.add(key)
        kept.append(speech)

    return kept, counts_by_congress


def stratified_holdout_ids(
    speech_ids: Sequence[str],
    congresses: Sequence[int],
    holdout_count: int,
    seed: int = 42,
) -> list[str]:
    """Pick exactly ``holdout_count`` speech IDs, allocated to each Congress in proportion to its size.

    When there are enough slots, every Congress with at least two speeches contributes at least one
    held-out speech, so the held-out set spans the full date range. Every Congress keeps at least
    one training speech.
    """
    by_congress: dict[int, list[str]] = {}
    for speech_id, congress in zip(speech_ids, congresses):
        by_congress.setdefault(int(congress), []).append(speech_id)

    total = len(speech_ids)
    capacity = {c: len(ids) - 1 for c, ids in by_congress.items()}
    if holdout_count > sum(capacity.values()):
        raise ValueError(f"Cannot hold out {holdout_count} speeches from a corpus of {total}.")

    rng = np.random.default_rng(seed)
    tie_break = {c: rng.random() for c in sorted(by_congress)}
    quotas = {c: holdout_count * len(ids) / total for c, ids in by_congress.items()}
    alloc = {c: min(capacity[c], math.floor(q)) for c, q in quotas.items()}
    eligible = [c for c in by_congress if capacity[c] > 0]
    if holdout_count >= len(eligible):
        for c in eligible:
            alloc[c] = max(alloc[c], 1)

    # Largest-remainder rounding so the allocation sums to exactly holdout_count. Each pass changes
    # at least one Congress, and the capacity check above guarantees a candidate exists.
    diff = holdout_count - sum(alloc.values())
    while diff > 0:
        grow = sorted((c for c in by_congress if alloc[c] < capacity[c]),
                      key=lambda c: (quotas[c] - alloc[c], tie_break[c]), reverse=True)
        for c in grow[:diff]:
            alloc[c] += 1
        diff -= len(grow[:diff])
    while diff < 0:
        shrink = sorted((c for c in by_congress if alloc[c] > 1),
                        key=lambda c: (alloc[c] - quotas[c], tie_break[c]), reverse=True)
        for c in shrink[:-diff]:
            alloc[c] -= 1
        diff += len(shrink[:-diff])

    held: list[str] = []
    for congress in sorted(by_congress):
        ids = by_congress[congress]
        picks = rng.choice(len(ids), size=alloc[congress], replace=False)
        held.extend(ids[i] for i in sorted(picks))
    return held


def create_speech_holdout(
    speeches: Sequence[SpeechRecord],
    holdout_count: int = DEFAULT_HOLDOUT_SPEECH_COUNT,
    seed: int = 42,
) -> tuple[list[str], list[SpeechRecord], list[SpeechRecord]]:
    if not speeches:
        raise ValueError("No speeches provided for holdout creation.")

    test_ids = stratified_holdout_ids(
        [speech.speech_id for speech in speeches],
        [speech.congress for speech in speeches],
        holdout_count=min(holdout_count, max(0, len(speeches) - 1)),
        seed=seed,
    )
    test_set = set(test_ids)
    train_speeches = [speech for speech in speeches if speech.speech_id not in test_set]
    eval_speeches = [speech for speech in speeches if speech.speech_id in test_set]
    return test_ids, train_speeches, eval_speeches


def chunk_speech_text(text: str, max_tokens: int, min_chunk_tokens: int = MIN_CHUNK_TOKENS) -> list[str]:
    tokens = (text or "").strip().split()
    if not tokens:
        return []

    chunks: list[str] = []
    for start in range(0, len(tokens), max_tokens):
        window = tokens[start : start + max_tokens]
        if len(window) < min_chunk_tokens:
            if chunks:
                chunks[-1] = chunks[-1] + " " + " ".join(window)
            else:
                chunks.append(" ".join(window))
            continue
        chunks.append(" ".join(window))

    return chunks


def chunk_token_ids(
    token_ids: Sequence[int],
    window: int,
    min_chunk_tokens: int = MIN_CHUNK_TOKENS,
    truncate: bool = False,
) -> tuple[list[list[int]], int]:
    """Split a tokenized speech into consecutive non-overlapping windows.

    Returns ``(chunks, dropped_token_count)``. Chunks shorter than ``min_chunk_tokens`` are dropped
    and counted. ``truncate=True`` reproduces the old loader: keep only the first window.
    """
    ids = list(token_ids)
    if truncate:
        return ([ids[:window]] if ids else []), max(0, len(ids) - window)

    chunks: list[list[int]] = []
    dropped = 0
    for start in range(0, len(ids), window):
        piece = ids[start : start + window]
        if len(piece) < min_chunk_tokens:
            dropped += len(piece)
        else:
            chunks.append(piece)
    return chunks, dropped


def build_chunked_dataset(speeches: Sequence[SpeechRecord], max_tokens: int) -> list[ChunkRecord]:
    chunks: list[ChunkRecord] = []
    for speech in speeches:
        subchunks = chunk_speech_text(speech.text, max_tokens=max_tokens)
        for idx, chunk in enumerate(subchunks):
            chunks.append(ChunkRecord(speech_id=speech.speech_id, congress=speech.congress, text=chunk, chunk_index=idx))
    return chunks


def validate_pipeline_counts(total_tokens_in: int, total_tokens_out: int, max_tokens: int) -> dict[str, float | int]:
    return {
        "total_tokens_in": total_tokens_in,
        "total_tokens_out": total_tokens_out,
        "token_ratio": float(total_tokens_out / max(1, total_tokens_in)),
        "max_tokens": max_tokens,
    }


def save_jsonl(path: str | Path, rows: Sequence[dict]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
