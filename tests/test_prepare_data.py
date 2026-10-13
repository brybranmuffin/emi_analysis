from __future__ import annotations

import csv
import gzip
import json
import re
from pathlib import Path

import pytest

from emi_analysis.prepare_data import main as prepare_main


def _norm(text: str) -> str:
    return " ".join(re.sub(r"[^\w\s]", "", text.lower()).split())


def _write_corpus(path: Path) -> None:
    rows = [{"speech_id": f"s{i}", "text": f"speech number {i} about topic {i} " * 5, "year": 1879} for i in range(1, 7)]
    rows += [{"speech_id": f"s{i}", "text": f"speech number {i} about topic {i} " * 5, "year": 1881} for i in range(7, 13)]
    # s13 is a near-duplicate of s1 (case and punctuation differ); s14 has no text.
    rows.append({"speech_id": "s13", "text": rows[0]["text"].upper().replace("SPEECH", "SPEECH,"), "year": 1879})
    rows.append({"speech_id": "s14", "text": "   ", "year": 1881})
    with gzip.open(path, "wt", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["speech_id", "text", "year"])
        writer.writeheader()
        writer.writerows(rows)


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


@pytest.fixture()
def corpus(tmp_path: Path) -> tuple[Path, Path, Path]:
    corpus_path = tmp_path / "combined_congress.csv.gzip"
    _write_corpus(corpus_path)
    general_path = tmp_path / "general.txt"
    general_path.write_text("\n".join(f"general domain paragraph {i} " * 10 for i in range(5)), encoding="utf-8")
    return corpus_path, general_path, tmp_path / "out"


def _run(corpus_path: Path, general_path: Path, out_root: Path, *extra: str) -> dict:
    return prepare_main(
        [
            "--input", str(corpus_path),
            "--out-root", str(out_root),
            "--holdout-count", "2",
            "--general-source", str(general_path),
            "--general-count", "2",
            *extra,
        ]
    )


def test_prepare_writes_disjoint_deduplicated_splits(corpus):
    corpus_path, general_path, out_root = corpus
    summary = _run(corpus_path, general_path, out_root)

    train = _read_jsonl(out_root / "train.jsonl")
    heldout = _read_jsonl(out_root / "heldout" / "congressional.jsonl")
    heldout_ids = (out_root / "heldout" / "speech_ids.txt").read_text().split()

    assert summary["input_rows"] == 14
    assert summary["dropped_empty_text"] == 1
    assert sorted(heldout_ids) == sorted(row["speech_id"] for row in heldout)
    assert len(heldout_ids) == 2
    assert {row["congress"] for row in heldout} == {46, 47}
    # 13 usable speeches - 2 held out - 1 near-duplicate of s1/s13.
    assert len(train) == 10
    assert summary["train_speeches"] == 10
    assert summary["dedup_removed_total"] == 1
    assert not set(heldout_ids) & {row["speech_id"] for row in train}
    assert summary["heldout_ids_in_train"] == 0
    all_texts = [_norm(row["text"]) for row in train + heldout]
    assert len(all_texts) == len(set(all_texts))
    assert (out_root / "processed" / "dedup_report.csv").exists()
    assert len(_read_jsonl(out_root / "heldout" / "general.jsonl")) == 2


def test_prepare_reuses_existing_holdout_ids(corpus):
    corpus_path, general_path, out_root = corpus
    _run(corpus_path, general_path, out_root, "--seed", "1")
    first_ids = (out_root / "heldout" / "speech_ids.txt").read_text()

    _run(corpus_path, general_path, out_root, "--seed", "99")

    assert (out_root / "heldout" / "speech_ids.txt").read_text() == first_ids


def test_prepare_reports_missing_columns(corpus, tmp_path):
    _, general_path, out_root = corpus
    bad = tmp_path / "bad.csv"
    bad.write_text("id,body\n1,hello\n", encoding="utf-8")
    with pytest.raises(ValueError, match="speech_id"):
        _run(bad, general_path, out_root)
