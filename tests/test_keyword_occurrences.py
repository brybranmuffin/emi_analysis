from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from emi_analysis.emi_common import ERAS, era_of, n_chunks_150, word_stats
from emi_analysis.keyword_occurrences import main as occurrences_main

from test_train import _make_run_inputs, _write_jsonl


def _corpus(tmp_path: Path) -> tuple[Path, list[dict], list[dict]]:
    """Two eras of training speeches plus held-out speeches that also contain every keyword."""
    config_path = _make_run_inputs(tmp_path, "bert")
    train = []
    for era_congress in (50, 60):  # eras 0 and 1
        for i in range(30):
            train.append({"speech_id": f"t{era_congress}-{i}", "congress": era_congress,
                          "text": f"The Evidence shows the study. I believe the evidence, {i} evidence-based. I feel it."})
    train += [{"speech_id": f"d{i}", "congress": 50, "text": "the data say so"} for i in range(2)]  # 'data': 2 in era 0
    heldout = [{"speech_id": f"h{i}", "congress": 50, "text": "evidence data study believe feel heart"} for i in range(5)]
    _write_jsonl(tmp_path / "data" / "train.jsonl", train)
    _write_jsonl(tmp_path / "data" / "heldout" / "congressional.jsonl", heldout)
    return config_path, train, heldout


def _run(tmp_path: Path, config_path: Path, *extra: str) -> Path:
    summary = occurrences_main(["--config", str(config_path), "--per-era", "5", "--unbalanced", "8", "--num-proc", "1",
                                "--output-root", str(tmp_path / "occ"), *extra])
    return Path(summary["output_dir"])


def test_sample_is_era_balanced_one_per_speech_and_excludes_heldout(tmp_path):
    config_path, train, heldout = _corpus(tmp_path)
    out = _run(tmp_path, config_path)
    occ = np.load(out / "occurrences.npz")
    words = json.loads((out / "keywords.json").read_text())["words"]
    texts = [r["text"] for r in train + heldout]
    ev = words.index("evidence")

    balanced = occ["rank_balanced"] >= 0
    for era in (0, 1):
        assert int((balanced & (occ["keyword"] == ev) & (occ["era"] == era)).sum()) == 5
    assert int(((occ["rank_unbalanced"] >= 0) & (occ["keyword"] == ev)).sum()) == 8
    assert (occ["row"] < len(train)).all()  # held-out speeches never contribute
    pairs = set(zip(occ["row"].tolist(), occ["keyword"].tolist()))
    assert len(pairs) == len(occ["row"])  # at most one occurrence per speech per keyword
    for row, kw, s, e in zip(occ["row"], occ["keyword"], occ["start"], occ["end"]):
        assert texts[row][s:e].lower() == words[kw]

    meta = json.loads((out / "meta.json").read_text())
    data = words.index("data")
    assert int((balanced & (occ["keyword"] == data)).sum()) == 2
    assert any(s["word"] == "data" and s["era"] == 0 and s["available"] == 2 for s in meta["shortfalls"])
    assert "data" in meta["flagged_rare"]
    stats = np.load(out / "speech_stats.npz")
    assert stats["heldout"].sum() == len(heldout) and stats["heldout"][-len(heldout):].all()
    assert stats["word_count"][-1] == 6


def test_sample_is_deterministic_and_prefix_ranked(tmp_path):
    config_path, _, _ = _corpus(tmp_path)
    first = np.load(_run(tmp_path, config_path, "--tag", "a") / "occurrences.npz")
    again = np.load(_run(tmp_path, config_path, "--tag", "b") / "occurrences.npz")
    for key in first.files:
        assert np.array_equal(first[key], again[key])
    small = np.load(_run(tmp_path, config_path, "--tag", "c", "--per-era", "3") / "occurrences.npz")

    def chosen(occ, limit):
        keep = (occ["rank_balanced"] >= 0) & (occ["rank_balanced"] < limit)
        return set(zip(occ["row"][keep].tolist(), occ["keyword"][keep].tolist()))

    assert chosen(small, 3) == chosen(first, 3)


def test_word_stats_match_paper_rules():
    from wordfreq import top_n_list

    top100 = set(top_n_list("en", 100))
    words = ["The", "cat", "sat", "on", "the", "mat"]
    count, frac = word_stats("The cat .. sat -- on __ the   mat", top100)
    assert count == 6
    assert frac == sum(w in top100 for w in words) / 6  # case-sensitive like the paper: "The" is not counted
    assert frac < sum(w.lower() in top100 for w in words) / 6


def test_chunk_count_merges_short_last_chunk():
    assert n_chunks_150(np.array([10, 150, 199, 200, 301, 350])).tolist() == [1, 1, 1, 2, 2, 3]


def test_era_boundaries():
    assert ERAS[0] == (46, 54) and ERAS[-1] == (109, 117) and len(ERAS) == 8
    assert era_of(np.array([45, 46, 54, 55, 117, 118])).tolist() == [-1, 0, 0, 1, 7, -1]
