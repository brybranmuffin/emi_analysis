from __future__ import annotations

import csv
import json
import random
from pathlib import Path

import numpy as np
import pytest

from emi_analysis.collect_activations import main as collect_main
from emi_analysis.emi_ablations import main as ablations_main
from emi_analysis.emi_score import main as score_main
from emi_analysis.emi_validate import congress_of_year, excerpt_targets, main as validate_main
from emi_analysis.keyword_activations import main as keyword_main
from emi_analysis.keyword_occurrences import main as occurrences_main

from test_train import WORDS, _make_run_inputs, _write_jsonl


def _read(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


@pytest.fixture(scope="module")
def pipeline(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("emi")
    config_path = _make_run_inputs(tmp, "bert", max_length=32)
    rng = random.Random(9)

    def speech(i):
        return " ".join(rng.choice(WORDS) for _ in range(rng.randint(12, 60)))

    _write_jsonl(tmp / "data" / "train.jsonl", [{"speech_id": f"s{i}", "congress": 46 + i % 72, "text": speech(i)} for i in range(120)])
    _write_jsonl(tmp / "data" / "heldout" / "congressional.jsonl",
                 [{"speech_id": f"h{i}", "congress": 46 + (i * 5) % 72, "text": speech(i)} for i in range(40)])
    inputs = ["--input", str(tmp / "data" / "train.jsonl"), "--input", str(tmp / "data" / "heldout" / "congressional.jsonl")]
    common = ["--model", "bert", "--config", str(config_path), "--base"]
    act = Path(collect_main([*common, *inputs, "--tag", "corpus", "--output-root", str(tmp / "act"), "--dtype", "float32",
                             "--precision", "fp32"])["output_dir"])
    occ = Path(occurrences_main(["--config", str(config_path), *inputs, "--per-era", "6", "--unbalanced", "10", "--num-proc", "1",
                                 "--output-root", str(tmp / "occ")])["output_dir"])
    kw = Path(keyword_main([*common, "--occurrences", str(occ), "--precision", "fp32", "--output-root", str(tmp / "kw")])["output_dir"])
    scores = Path(score_main(["--model", "bert", "--activations", str(act), "--keywords", str(kw), "--occurrences", str(occ),
                              "--n-per-era", "4", "--output-root", str(tmp / "results"), "--n-boot", "20"])["output_dir"])
    return {"tmp": tmp, "config": config_path, "act": act, "occ": occ, "kw": kw, "scores": scores, "common": common}


def test_ablation_table_has_every_variant(pipeline):
    out = Path(ablations_main(["--scores", str(pipeline["scores"]), "--output-root", str(pipeline["tmp"] / "abl")])["output_dir"])
    rows = {r["variant"]: r for r in _read(out / "ablations.csv")}
    words = json.loads((pipeline["occ"] / "keywords.json").read_text())["words"]
    expected = {"context/alone", "sampling/unbalanced", "scoring/projection", "scoring/no_length_adjust",
                "scoring/no_centering", "series/chunk_weighted", "n_per_era/25", "n_per_era/50", "n_per_era/200"}
    assert expected <= set(rows)
    assert {f"era_poles/{e}" for e in range(8)} <= set(rows)
    primary = json.loads((pipeline["scores"] / "meta.json").read_text())["primary_layer"]
    assert np.isclose(float(rows[f"layers/{primary}"]["pearson_speech"]), 1.0)
    assert float(rows["series/chunk_weighted"]["pearson_speech"]) == pytest.approx(1.0)
    assert {r["variant"] for r in _read(out / "loo_keywords.csv")} <= {f"loo/{w}" for w in words}
    cos = _read(out / "era_pole_cosines.csv")
    assert {r["pole"] for r in cos} == {"evidence", "intuition", "difference"}
    assert any(r["era_a"] == "pooled" for r in cos)
    assert (out / "ablation_series.csv").exists()


def test_loo_drops_one_keyword_each(pipeline):
    out = Path(ablations_main(["--scores", str(pipeline["scores"]), "--output-root", str(pipeline["tmp"] / "abl2")])["output_dir"])
    loo = _read(out / "loo_keywords.csv")
    words = json.loads((pipeline["occ"] / "keywords.json").read_text())["words"]
    kw_vectors = np.load(pipeline["scores"] / f"poles_layer_{json.loads((pipeline['scores'] / 'meta.json').read_text())['primary_layer']:02d}.npz")["keyword_vectors"]
    with_vectors = [w for w, v in zip(words, kw_vectors) if np.isfinite(v).all()]
    assert sorted(r["variant"] for r in loo) == sorted(f"loo/{w}" for w in with_vectors)


def test_subset_collection_scores_against_corpus_occurrences(pipeline):
    tmp = pipeline["tmp"]
    held = Path(collect_main([*pipeline["common"], "--input", str(tmp / "data" / "heldout" / "congressional.jsonl"), "--tag", "heldout",
                              "--output-root", str(tmp / "act"), "--dtype", "float32", "--precision", "fp32"])["output_dir"])
    sub = Path(score_main(["--model", "bert", "--activations", str(held), "--keywords", str(pipeline["kw"]),
                           "--occurrences", str(pipeline["occ"]), "--n-per-era", "4", "--layers", "2",
                           "--output-root", str(tmp / "results_sub"), "--n-boot", "20"])["output_dir"])
    assert np.load(sub / "layer_02_ev.npy").shape == (40,)
    out = Path(ablations_main(["--scores", str(pipeline["scores"]), "--compare", str(sub), "--output-root", str(tmp / "abl3")])["output_dir"])
    rows = {r["variant"]: r for r in _read(out / "ablations.csv")}
    assert any(v.startswith("compare/") for v in rows)


def test_w2v_year_to_congress_mapping():
    assert congress_of_year(np.array([1879, 1881, 1882, 2021])).tolist() == [46, 47, 47, 117]


def test_excerpt_targets_average_raters(tmp_path):
    path = tmp_path / "ratings.csv"
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["ResponseId", "evidence_free", "evidence_based", "speech_id", "text", "item_id", "starting_year",
                         "evidence_minus_intuition_score"])
        writer.writerow(["r1", 1, 5, 10, "text one", 1, 1881, 0.5])
        writer.writerow(["r2", 3, 4, 10, "text one", 1, 1881, 0.5])
        writer.writerow(["r1", 4, 2, 11, "text two", 2, 2001, -0.2])
    targets = {t["speech_id"]: t for t in excerpt_targets(path)}
    assert targets["item1"]["target"] == pytest.approx(((5 - 1) + (4 - 3)) / 2) and targets["item1"]["n_raters"] == 2
    assert targets["item2"]["congress"] == 107 and targets["item2"]["w2v_emi"] == pytest.approx(-0.2)


def test_validation_end_to_end(pipeline, tmp_path):
    ratings = tmp_path / "ratings.csv"
    rng = random.Random(2)
    with ratings.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["ResponseId", "evidence_free", "evidence_based", "speech_id", "text", "item_id", "starting_year",
                         "evidence_minus_intuition_score"])
        for item in range(12):
            text = " ".join(rng.choice(WORDS) for _ in range(40))
            for rater in range(2):
                writer.writerow([f"r{rater}", rng.randint(1, 5), rng.randint(1, 5), 100 + item, text, item, 1881 + 2 * item, rng.random()])
    w2v = tmp_path / "w2v.csv"
    w2v.write_text("year,evidence_minus_intuition_score,lower_bound,upper_bound\n"
                   + "".join(f"{1879 + 2 * i},{rng.random()},0,1\n" for i in range(72)))
    excerpts = tmp_path / "excerpts.jsonl"
    validate_main(["export", "--prolific", str(ratings), "--output", str(excerpts)])
    exc_act = Path(collect_main([*pipeline["common"], "--input", str(excerpts), "--tag", "prolific",
                                 "--output-root", str(tmp_path / "act"), "--dtype", "float32", "--precision", "fp32"])["output_dir"])
    report = validate_main(["score", "--scores", str(pipeline["scores"]), "--excerpt-activations", str(exc_act),
                            "--prolific", str(ratings), "--w2v", str(w2v), "--output-root", str(tmp_path / "val")])
    assert {"trend", "human"} <= set(report)
    assert report["human"]["n_items"] == 12 and "pearson" in report["human"] and "w2v_pearson" in report["human"]
    assert report["trend"]["n_congresses"] > 0
