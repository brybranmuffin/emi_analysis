from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

from sklearn.model_selection import train_test_split

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


def create_speech_holdout(
    speeches: Sequence[SpeechRecord],
    holdout_count: int = DEFAULT_HOLDOUT_SPEECH_COUNT,
    seed: int = 42,
) -> tuple[list[str], list[SpeechRecord], list[SpeechRecord]]:
    if not speeches:
        raise ValueError("No speeches provided for holdout creation.")

    congresses = sorted({speech.congress for speech in speeches})
    test_ids: list[str] = []

    for congress in congresses:
        items = [speech for speech in speeches if speech.congress == congress]
        ids = [speech.speech_id for speech in items]
        if len(ids) <= 1:
            continue
        train_ids, heldout_ids = train_test_split(
            ids,
            test_size=min(len(ids), max(1, int(round(len(ids) * (holdout_count / max(1, len(speeches))))))),
            random_state=seed + congress,
            shuffle=True,
        )
        test_ids.extend(heldout_ids)

    if len(test_ids) > holdout_count:
        test_ids = test_ids[:holdout_count]
    elif len(test_ids) < holdout_count:
        remaining = [speech.speech_id for speech in speeches if speech.speech_id not in test_ids]
        test_ids.extend(remaining[: holdout_count - len(test_ids)])

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
