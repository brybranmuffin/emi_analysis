from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import pytest
import torch

from emi_analysis.collect_activations import main as collect_main
from emi_analysis.sae import TopKSAE, load_sae
from emi_analysis.train import collate, load_model_and_tokenizer, parse_config
from emi_analysis.train_sae import era_balanced_rows, find_layer_module, main as train_sae_main

from test_train import WORDS, _make_run_inputs, _write_jsonl


def _data(n: int = 2048, d: int = 16, seed: int = 0) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    atoms = torch.randn(64, d, generator=g)
    codes = torch.zeros(n, 64)
    idx = torch.randint(0, 64, (n, 3), generator=g)
    codes.scatter_(1, idx, torch.rand(n, 3, generator=g) + 0.5)
    return codes @ atoms + 5.0


def test_topk_keeps_exactly_k():
    sae = TopKSAE(16, 64, k=4)
    x = _data(32)
    sae.set_normalization(x.mean(0), x)
    f = sae.encode(x)
    assert f.shape == (32, 64)
    assert ((f > 0).sum(dim=1) <= 4).all() and (f >= 0).all()


def test_training_reduces_reconstruction_error():
    torch.manual_seed(0)
    x = _data()
    sae = TopKSAE(16, 64, k=4)
    sae.set_normalization(x.mean(0), x)
    before = sae.fvu(x)
    opt = torch.optim.Adam(sae.parameters(), lr=3e-3)
    dead = torch.zeros(64, dtype=torch.bool)
    for _ in range(300):
        loss, _ = sae.loss(x[torch.randint(0, len(x), (256,))], dead)
        opt.zero_grad()
        loss.backward()
        opt.step()
        sae.normalize_decoder()
    assert sae.fvu(x) < 0.5 * before
    assert torch.allclose(sae.W_dec.norm(dim=1), torch.ones(64), atol=1e-5)


def test_save_load_roundtrip(tmp_path):
    x = _data(64)
    sae = TopKSAE(16, 64, k=4)
    sae.set_normalization(x.mean(0), x)
    sae.save(tmp_path / "sae.pt", {"layer": 3, "kind": "token"})
    loaded, meta = load_sae(tmp_path / "sae.pt", "cpu")
    assert meta["layer"] == 3 and meta["name"] == tmp_path.name
    assert torch.equal(loaded.encode(x), sae.encode(x))


@pytest.mark.parametrize("family", ["bert", "gpt2", "qwen3"])
def test_find_layer_module_matches_hidden_state(tmp_path, family, tiny_qwen3_path):
    overrides = {"model_name": tiny_qwen3_path} if family == "qwen3" else {}
    config = parse_config(str(_make_run_inputs(tmp_path, family, **overrides)))
    model, tokenizer = load_model_and_tokenizer(config, config.model_name, torch.device("cpu"))
    model.eval()
    batch = collate([tokenizer("the committee report", add_special_tokens=False)["input_ids"]], tokenizer.pad_token_id)
    with torch.no_grad():
        states = model.base_model(**batch, output_hidden_states=True).hidden_states
    for layer in (0, 1, len(states) - 1):
        module = find_layer_module(model.base_model, batch, layer)
        captured = {}
        handle = module.register_forward_hook(lambda m, i, o: captured.setdefault("out", o[0] if isinstance(o, tuple) else o))
        with torch.no_grad():
            model.base_model(**batch)
        handle.remove()
        assert torch.equal(captured["out"], states[layer])


def test_era_balanced_rows_excludes_heldout_and_balances_tokens():
    congress = np.array([50] * 50 + [60] * 10 + [50] * 5)
    counts = np.array([10] * 50 + [50] * 10 + [10] * 5)
    heldout = np.r_[np.zeros(60, dtype=bool), np.ones(5, dtype=bool)]
    rows, per_era = era_balanced_rows(congress, counts, heldout, total_tokens=400, seed=0)
    assert not heldout[rows].any()
    assert per_era[0] >= 200 and per_era[1] >= 200
    assert per_era[0] < 200 + 10 and per_era[1] < 200 + 50


def _corpus(tmp_path: Path, family: str) -> tuple[Path, Path]:
    config_path = _make_run_inputs(tmp_path, family, max_length=32)
    rng = random.Random(4)
    rows = [{"speech_id": f"s{i}", "congress": 46 + 9 * (i % 8), "text": " ".join(rng.choice(WORDS) for _ in range(rng.randint(5, 60)))}
            for i in range(64)]
    _write_jsonl(tmp_path / "data" / "train.jsonl", rows)
    inputs = ["--input", str(tmp_path / "data" / "train.jsonl"), "--input", str(tmp_path / "data" / "heldout" / "congressional.jsonl")]
    act = Path(collect_main(["--model", family, "--config", str(config_path), "--base", *inputs, "--tag", "corpus",
                             "--output-root", str(tmp_path / "act"), "--dtype", "float32", "--precision", "fp32"])["output_dir"])
    return config_path, act


def test_token_training_end_to_end(tmp_path):
    config_path, act = _corpus(tmp_path, "bert")
    summary = train_sae_main(["--model", "bert", "--config", str(config_path), "--kind", "token", "--layer", "2",
                              "--activations", str(act), "--k", "4", "--expansion", "2", "--tokens", "3000",
                              "--batch-size", "128", "--buffer-tokens", "1000", "--output-root", str(tmp_path / "sae")])
    out = Path(summary["output_dir"])
    metrics = json.loads((out / "metrics.json").read_text())
    sae, meta = load_sae(out / "sae.pt", "cpu")
    assert meta["layer"] == 2 and meta["kind"] == "token" and sae.k == 4
    assert metrics["tokens_trained"] >= 3000 and 0 <= metrics["heldout_fvu"] < 1.5
    assert set(metrics["tokens_per_era"]) == {str(e) for e in range(8)}
    assert "spliced_loss" in metrics and "original_loss" in metrics
    assert (out / "history.csv").exists()


def test_speech_training_end_to_end(tmp_path):
    config_path, act = _corpus(tmp_path, "gpt2")
    summary = train_sae_main(["--model", "gpt2", "--config", str(config_path), "--kind", "speech", "--layer", "1",
                              "--activations", str(act), "--k", "4", "--expansion", "2", "--samples", "2000",
                              "--batch-size", "64", "--output-root", str(tmp_path / "sae")])
    out = Path(summary["output_dir"])
    sae, meta = load_sae(out / "sae.pt", "cpu")
    assert meta["kind"] == "speech" and meta["layer"] == 1
    metrics = json.loads((out / "metrics.json").read_text())
    assert metrics["samples_trained"] >= 2000 and "heldout_fvu" in metrics


def test_keyword_activations_store_mean_of_piece_features(tmp_path):
    from emi_analysis.keyword_activations import main as keyword_main
    from emi_analysis.keyword_occurrences import main as occurrences_main

    config_path, act = _corpus(tmp_path, "bert")
    sae_dir = Path(train_sae_main(["--model", "bert", "--config", str(config_path), "--kind", "token", "--layer", "2",
                                   "--activations", str(act), "--k", "4", "--expansion", "2", "--tokens", "1000",
                                   "--batch-size", "128", "--buffer-tokens", "500", "--output-root", str(tmp_path / "sae")])["output_dir"])
    occ = Path(occurrences_main(["--config", str(config_path), "--per-era", "3", "--unbalanced", "4", "--num-proc", "1",
                                 "--input", str(tmp_path / "data" / "train.jsonl"),
                                 "--input", str(tmp_path / "data" / "heldout" / "congressional.jsonl"),
                                 "--output-root", str(tmp_path / "occ")])["output_dir"])
    kw = Path(keyword_main(["--model", "bert", "--config", str(config_path), "--base", "--occurrences", str(occ),
                            "--precision", "fp32", "--sae", str(sae_dir / "sae.pt"), "--output-root", str(tmp_path / "kw")])["output_dir"])
    features = np.load(kw / f"occ_sae_{sae_dir.name}.npy")
    raw = np.load(kw / "occ_layer_02.npy")
    sae, _ = load_sae(sae_dir / "sae.pt", "cpu")
    assert features.shape == (len(raw), sae.n_features) and np.isfinite(features).all()
    alone = np.load(kw / f"alone_sae_{sae_dir.name}.npy")
    assert alone.shape == (len(json.loads((occ / "keywords.json").read_text())["words"]), sae.n_features)
    config = parse_config(str(config_path))
    _, tokenizer = load_model_and_tokenizer(config, config.model_name, torch.device("cpu"))
    words = json.loads((occ / "keywords.json").read_text())["words"]
    pieces = np.asarray([len(tokenizer(w, add_special_tokens=False)["input_ids"]) for w in words])
    keyword = np.load(occ / "occurrences.npz")["keyword"]
    single, multi = pieces[keyword] == 1, pieces[keyword] > 1
    assert single.any() or multi.any()
    # One piece: mean of SAE(piece) is SAE(the state). Several pieces: it is the mean of per-piece features,
    # which differs from SAE(mean of the pieces' states) because TopK is nonlinear.
    with torch.no_grad():
        encoded_mean = sae.encode(torch.from_numpy(raw)).numpy()
    if single.any():
        assert np.allclose(features[single], encoded_mean[single], atol=1e-5)
    if multi.any():
        assert not np.allclose(features[multi], encoded_mean[multi], atol=1e-5)
