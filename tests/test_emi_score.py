from __future__ import annotations

import csv
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import zscore

from emi_analysis.collect_activations import main as collect_main
from emi_analysis.emi_score import (congress_series, cosines, emi_from_cosines, keyword_vectors, layer_auc, main as score_main,
                                    poles, speech_layer_auc, standardize, token_weighted_mean)
from emi_analysis.emi_select_layer import main as select_main
from emi_analysis.keyword_activations import main as keyword_main
from emi_analysis.keyword_occurrences import main as occurrences_main

from test_train import WORDS, _make_run_inputs, _write_jsonl


def test_token_weighted_mean_equals_mean_over_tokens():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(50, 4)).astype(np.float16)
    counts = rng.integers(1, 30, size=50)
    expected = (counts[:, None] * x.astype(np.float64)).sum(0) / counts.sum()
    assert np.allclose(token_weighted_mean(x, counts, block=7), expected)
    rows = np.arange(10, 40)
    sub = (counts[rows, None] * x[rows].astype(np.float64)).sum(0) / counts[rows].sum()
    assert np.allclose(token_weighted_mean(x, counts, rows=rows, block=7), sub)


def test_poles_weight_eras_equally():
    a, b = np.array([1.0, 0.0]), np.array([0.0, 1.0])
    occ = np.stack([a] * 10 + [b]).astype(np.float32)
    keyword = np.zeros(11, dtype=np.int64)
    era = np.array([0] * 10 + [1])
    use = np.ones(11, dtype=bool)
    balanced, missing = keyword_vectors(occ, keyword, era, 1, use)
    assert np.allclose(balanced[0], (a + b) / 2) and missing == []
    unbalanced, _ = keyword_vectors(occ, keyword, era, 1, use, balanced=False)
    assert np.allclose(unbalanced[0], (10 * a + b) / 11)


def test_missing_keyword_excluded_and_reported():
    occ = np.array([[1.0, 0.0], [0.0, 1.0], [5.0, 5.0], [np.nan, np.nan]], dtype=np.float32)
    keyword, era = np.array([0, 1, 2, 3]), np.zeros(4, dtype=np.int64)
    use = np.array([True, True, False, True])  # keyword 2 not used; keyword 3's only span has no token
    kw, missing = keyword_vectors(occ, keyword, era, 4, use)
    assert missing == [2, 3] and np.isnan(kw[2]).all()
    mu = np.array([0.5, 0.5])
    p_ev, p_int = poles(kw, np.array([0, 1, 0, 1]), mu)
    assert np.allclose(p_ev, [0.5, -0.5]) and np.allclose(p_int, [-0.5, 0.5])


def _paper_emi(ev, intu, wc):
    """compute_sbert_avg_lexicon.py's length_adjustment_bin + evidence_minus_intuition_score, verbatim logic."""
    df = pd.DataFrame({"avg_evidence_score": ev, "avg_intuition_score": intu, "length": wc})
    bins = range(10, df["length"].max() + 10, 10)
    df["length_bin"] = pd.cut(df["length"], bins=bins)
    df["evidence_adj"] = df["avg_evidence_score"] - df.groupby("length_bin", observed=False)["avg_evidence_score"].transform("mean")
    df["intuition_adj"] = df["avg_intuition_score"] - df.groupby("length_bin", observed=False)["avg_intuition_score"].transform("mean")
    z = df[["evidence_adj", "intuition_adj"]].apply(zscore)
    return (z["evidence_adj"] - z["intuition_adj"]).to_numpy()


def test_emi_matches_paper_formula_on_toy_data():
    rng = np.random.default_rng(1)
    wc = rng.integers(11, 95, size=300)
    ev, intu = rng.normal(size=300), rng.normal(size=300) + wc / 100
    emi, params = emi_from_cosines(ev, intu, wc, np.ones(300, dtype=bool))
    assert np.allclose(emi, _paper_emi(ev, intu, wc))
    assert {"ev", "int"} <= set(params)


def test_unscorable_speeches_get_no_score_and_single_speech_bin():
    ev, intu = np.array([0.1, 0.2, 0.3, 0.5]), np.array([0.0, 0.1, 0.0, 0.2])
    wc = np.array([5, 15, 16, 200])  # first is too short; 200 sits alone in its bin
    emi, _ = emi_from_cosines(ev, intu, wc, np.array([False, True, True, True]))
    assert np.isnan(emi[0]) and np.isfinite(emi[1:]).all()
    z, _ = standardize(ev, wc, np.array([False, True, True, True]))
    assert z[3] == 0.0 or np.isfinite(z[3])


def test_series_handles_empty_congress():
    rng = np.random.default_rng(2)
    congress = np.repeat([46, 47, 48], 100)
    emi = rng.normal(size=300)
    emi[100:200] = np.nan
    rows = {r["congress"]: r for r in congress_series(emi, congress, n_boot=200)}
    assert rows[47]["n"] == 0 and np.isnan(rows[47]["mean"])
    assert rows[46]["lower"] <= rows[46]["mean"] <= rows[46]["upper"]
    weighted = {r["congress"]: r for r in congress_series(emi, congress, weights=np.r_[np.ones(50), np.zeros(250)], n_boot=200)}
    assert np.isclose(weighted[46]["mean"], np.nanmean(emi[:50]))


def test_layer_auc_separates_planted_axis():
    rng = np.random.default_rng(3)
    n_kw, per = 10, 30
    keyword = np.repeat(np.arange(n_kw), per)
    pole = (keyword >= 5).astype(np.int8)
    axis = rng.normal(size=16)
    occ = rng.normal(size=(n_kw * per, 16)) + np.where(pole[:, None] == 0, 1.0, -1.0) * axis
    era = rng.integers(0, 8, size=len(keyword))
    use = np.ones(len(keyword), dtype=bool)
    assert layer_auc(occ, keyword, pole, era, np.zeros(16), use, n_kw) > 0.9
    random_occ = rng.normal(size=occ.shape)
    assert 0.3 < layer_auc(random_occ, keyword, pole, era, np.zeros(16), use, n_kw) < 0.7


def _planted_speeches(rng, signal: float = 1.0, n_kw: int = 10, per: int = 30, dim: int = 16):
    """One occurrence per speech; each speech vector and its occurrence vector lean toward their keyword's pole."""
    keyword = np.repeat(np.arange(n_kw), per)
    pole = (keyword >= n_kw // 2).astype(np.int8)
    axis = rng.normal(size=dim)
    lean = np.where(pole[:, None] == 0, 1.0, -1.0) * axis
    occ = rng.normal(size=(len(keyword), dim)) + 2 * lean
    speeches = rng.normal(size=(len(keyword), dim)) * 3 + signal * lean
    return {"source": speeches, "occ": occ, "row": np.arange(len(keyword)), "keyword": keyword, "pole": pole,
            "era": rng.integers(0, 8, size=len(keyword)), "mu": np.zeros(dim), "use": np.ones(len(keyword), dtype=bool),
            "n_keywords": n_kw, "ok": np.ones(len(keyword), dtype=bool)}


def test_speech_auc_scores_whole_speeches_not_occurrences():
    rng = np.random.default_rng(6)
    strong = speech_layer_auc(**_planted_speeches(rng, signal=1.0))
    assert strong["auc"] > 0.9 and strong["n_speeches"] == 300 and strong["n_keywords"] == 10
    # Occurrences separate perfectly but the speeches carry no signal: the speech AUC must say so.
    flat = speech_layer_auc(**_planted_speeches(rng, signal=0.0))
    assert 0.3 < flat["auc"] < 0.7
    assert layer_auc(*(_planted_speeches(rng, signal=0.0)[k] for k in ("occ", "keyword", "pole", "era", "mu", "use", "n_keywords"))) > 0.9


def test_primary_layer_skips_the_embedding_layer():
    from emi_analysis.emi_score import choose_primary

    rows = [{"layer": 0, "loo_speech_auc": 0.70}, {"layer": 1, "loo_speech_auc": 0.60}, {"layer": 2, "loo_speech_auc": 0.65},
            {"layer": 3, "loo_speech_auc": 0.65}]
    assert choose_primary(rows, [0, 1, 2, 3]) == 3  # layer 0 is best but excluded; the tie goes deeper
    assert choose_primary(rows[:1], [0]) == 0  # an SAE result has only its own layer
    assert choose_primary([{"layer": "0", "loo_speech_auc": "0.7"}, {"layer": "1", "loo_speech_auc": "nan"}], [0, 1]) == 1


def test_speech_auc_skips_pole_speeches_and_unscorable_ones():
    rng = np.random.default_rng(7)
    data = _planted_speeches(rng)
    # Speech 0 also supplies a pole occurrence of keyword 9 (it holds two sampled keywords): never scored.
    data["row"] = np.r_[data["row"], 0]
    data["keyword"] = np.r_[data["keyword"], 9]
    data["pole"] = np.r_[data["pole"], 1]
    data["era"] = np.r_[data["era"], 0]
    data["use"] = np.r_[data["use"], True]
    data["occ"] = np.vstack([data["occ"], data["occ"][-1:]])
    data["ok"] = data["ok"].copy()
    data["ok"][1] = False  # too short for the paper's filters
    out = speech_layer_auc(**data)
    assert out["n_speeches"] == 300 - 2 and out["n_excluded_shared"] == 1
    # An occurrence outside `use` does not feed the poles, so its speech stays scorable.
    data["use"][-1] = False
    assert speech_layer_auc(**data)["n_speeches"] == 300 - 1


def test_speech_auc_applies_saved_standardization():
    rng = np.random.default_rng(8)
    data = _planted_speeches(rng)
    wc = np.where(data["pole"] == 0, 100, 20)  # evidence speeches sit in bin 8, intuition speeches in bin 0
    plain = speech_layer_auc(**data, word_count=wc)
    assert plain["auc"] > 0.9
    zero = {"adjust": True, "bin_means": {0: 0.0, 8: 0.0}, "mean": 0.0, "sd": 1.0}
    assert np.isclose(speech_layer_auc(**data, word_count=wc, params={"ev": zero, "int": zero})["auc"], plain["auc"])
    # The corpus says bin-8 speeches have a far higher evidence cosine: once that is subtracted, the ranking flips.
    ev = {"adjust": True, "bin_means": {0: 0.0, 8: 100.0}, "mean": 0.0, "sd": 1.0}
    assert speech_layer_auc(**data, word_count=wc, params={"ev": ev, "int": zero})["auc"] < 0.1


def test_end_to_end_raw_scoring_on_tiny_bert(tmp_path):
    config_path = _make_run_inputs(tmp_path, "bert", max_length=32)
    rng = random.Random(5)
    rows = [{"speech_id": f"s{i}", "congress": 46 + (i % 72),
             "text": " ".join(rng.choice(WORDS) for _ in range(rng.randint(3, 60)))} for i in range(80)]
    _write_jsonl(tmp_path / "data" / "train.jsonl", rows)
    inputs = ["--input", str(tmp_path / "data" / "train.jsonl"), "--input", str(tmp_path / "data" / "heldout" / "congressional.jsonl")]
    act = Path(collect_main(["--model", "bert", "--config", str(config_path), "--base", *inputs, "--tag", "corpus",
                             "--output-root", str(tmp_path / "act"), "--dtype", "float32", "--precision", "fp32"])["output_dir"])
    occ = Path(occurrences_main(["--config", str(config_path), *inputs, "--per-era", "4", "--unbalanced", "8",
                                 "--num-proc", "1", "--output-root", str(tmp_path / "occ")])["output_dir"])
    kw = Path(keyword_main(["--model", "bert", "--config", str(config_path), "--base", "--occurrences", str(occ),
                            "--precision", "fp32", "--output-root", str(tmp_path / "kw")])["output_dir"])
    summary = score_main(["--model", "bert", "--activations", str(act), "--keywords", str(kw), "--occurrences", str(occ),
                          "--representation", "raw", "--n-per-era", "4", "--min-per-era", "1", "--output-root", str(tmp_path / "results"),
                          "--n-boot", "50"])
    out = Path(summary["output_dir"])
    meta = json.loads((out / "meta.json").read_text())
    n = len(rows) + 6
    n_layers = json.loads((act / "meta.json").read_text())["n_layers"]
    assert 0 <= meta["primary_layer"] < n_layers
    emi = np.load(out / "layer_03_emi.npy")
    stats = np.load(occ / "speech_stats.npz")
    assert emi.shape == (n,)
    assert (np.isnan(emi) == ~((stats["word_count"] > 10) & (stats["top100_frac"] >= 0.05))).all()

    # One speech's evidence cosine recomputed by hand from the saved mean and poles.
    mu = np.load(out / "mu_layer_03.npy")
    p = np.load(out / "poles_layer_03.npz")
    x = np.load(act / "layer_03.npy")[7].astype(np.float64) - mu
    expected = x @ p["p_ev"] / (np.linalg.norm(x) * np.linalg.norm(p["p_ev"]))
    assert np.isclose(np.load(out / "layer_03_ev.npy")[7], expected, atol=1e-5)
    with (out / "series.csv").open() as handle:
        series = list(csv.DictReader(handle))
    assert {r["layer"] for r in series} == {str(i) for i in range(n_layers)}
    with (out / "layer_selection.csv").open() as handle:
        selection = list(csv.DictReader(handle))
    assert len(selection) == n_layers and {"loo_keyword_auc", "loo_speech_auc", "n_auc_speeches"} <= set(selection[0])
    assert "speech" in meta["primary_rule"]

    # Re-selecting from the saved results reproduces the same speech AUCs and keeps the old choice on record.
    meta["primary_layer"], meta["primary_rule"] = 0, "highest leave-one-keyword-out AUC (ties: deeper)"
    (out / "meta.json").write_text(json.dumps(meta))
    (out / "layer_selection.csv").write_text("layer,loo_keyword_auc,pole_cosine,missing_keywords\n" + "".join(
        f"{r['layer']},{r['loo_keyword_auc']},{r['pole_cosine']},{r['missing_keywords']}\n" for r in selection))
    result = select_main([str(out)])
    again = json.loads((out / "meta.json").read_text())
    with (out / "layer_selection.csv").open() as handle:
        reselected = list(csv.DictReader(handle))
    assert [r["loo_speech_auc"] for r in reselected] == [r["loo_speech_auc"] for r in selection]
    assert [r["loo_keyword_auc"] for r in reselected] == [r["loo_keyword_auc"] for r in selection]
    assert again["primary_layer"] == result[0]["primary_layer"] == summary["primary_layer"]
    assert again["previous_primary"] == {"primary_layer": 0, "primary_rule": "highest leave-one-keyword-out AUC (ties: deeper)"}
    assert "speech" in again["primary_rule"]


class _RowsOnly:
    """An occurrence array that may only be read by row index (as a memmap of 8,192 SAE features must be)."""

    def __init__(self, data):
        self.data, self.shape, self.read = data, data.shape, 0

    def __len__(self):
        return len(self.data)

    def __array__(self, *args, **kwargs):
        raise AssertionError("the whole occurrence array was loaded")

    def __getitem__(self, index):
        out = self.data[index]
        self.read += len(out)
        return out


def test_keyword_vectors_and_auc_read_only_used_rows():
    rng = np.random.default_rng(4)
    occ = _RowsOnly(rng.normal(size=(100, 8)))
    keyword = np.repeat(np.arange(4), 25)
    era = np.zeros(100, dtype=np.int64)
    use = np.zeros(100, dtype=bool)
    use[::5] = True
    kw, _ = keyword_vectors(occ, keyword, era, 4, use)
    assert np.isfinite(kw).all() and occ.read == use.sum()
    assert np.isfinite(layer_auc(occ, keyword, (keyword >= 2).astype(np.int8), era, np.zeros(8), use, 4))


def test_min_per_era_skips_thin_eras_and_drops_keywords_with_none():
    a, b = np.array([1.0, 0.0]), np.array([0.0, 1.0])
    occ = np.stack([a] * 10 + [b] + [b] * 3).astype(np.float32)
    keyword = np.array([0] * 11 + [1] * 3)
    era = np.array([0] * 10 + [1] + [2] * 3)
    use = np.ones(len(keyword), dtype=bool)
    kw, missing = keyword_vectors(occ, keyword, era, 2, use, min_per_era=5)
    assert np.allclose(kw[0], a)  # the single era-1 occurrence no longer counts as a whole era
    assert missing == [1] and np.isnan(kw[1]).all()  # keyword 1 has no era with 5 occurrences
    old, _ = keyword_vectors(occ, keyword, era, 2, use)
    assert np.allclose(old[0], (a + b) / 2)


def test_scoring_refuses_poles_without_keywords(tmp_path):
    import pytest

    config_path = _make_run_inputs(tmp_path, "bert", max_length=32)
    inputs = ["--input", str(tmp_path / "data" / "train.jsonl"), "--input", str(tmp_path / "data" / "heldout" / "congressional.jsonl")]
    act = Path(collect_main(["--model", "bert", "--config", str(config_path), "--base", *inputs, "--tag", "corpus",
                             "--output-root", str(tmp_path / "act"), "--dtype", "float32", "--precision", "fp32"])["output_dir"])
    occ = Path(occurrences_main(["--config", str(config_path), *inputs, "--per-era", "4", "--unbalanced", "8",
                                 "--num-proc", "1", "--output-root", str(tmp_path / "occ")])["output_dir"])
    kw = Path(keyword_main(["--model", "bert", "--config", str(config_path), "--base", "--occurrences", str(occ),
                            "--precision", "fp32", "--output-root", str(tmp_path / "kw")])["output_dir"])
    with pytest.raises(RuntimeError, match="min-per-era"):
        score_main(["--model", "bert", "--activations", str(act), "--keywords", str(kw), "--occurrences", str(occ),
                    "--layers", "1", "--output-root", str(tmp_path / "results")])
