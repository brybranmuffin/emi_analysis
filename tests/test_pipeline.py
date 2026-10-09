from __future__ import annotations

from emi_analysis.data_pipeline import chunk_speech_text


def test_chunking_preserves_tokens_without_truncation():
    text = " ".join(f"token{i}" for i in range(700))
    chunks = chunk_speech_text(text, max_tokens=256)
    total = sum(len(chunk.split()) for chunk in chunks)
    assert total == 700
    assert all(len(chunk.split()) <= 256 for chunk in chunks)


def test_short_trailing_chunk_is_merged_or_dropped_cleanly():
    text = " ".join(f"w{i}" for i in range(270))
    chunks = chunk_speech_text(text, max_tokens=256)
    assert sum(len(chunk.split()) for chunk in chunks) == 270
    assert len(chunks) >= 1
