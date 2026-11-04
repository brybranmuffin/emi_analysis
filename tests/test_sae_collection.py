from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from emi_analysis.collect_activations import main as collect_main
from emi_analysis.data_pipeline import chunk_token_ids
from emi_analysis.sae import load_sae
from emi_analysis.train import load_model_and_tokenizer, parse_config
from emi_analysis.train_sae import main as train_sae_main
from emi_analysis.verify_activations import main as verify_main

from test_sae import _corpus


def _train(tmp_path: Path, config_path: Path, act: Path, seed: int = 0) -> Path:
    return Path(train_sae_main(["--model", "bert", "--config", str(config_path), "--kind", "token", "--layer", "2",
                                "--activations", str(act), "--k", "4", "--expansion", "2", "--tokens", "1500", "--seed", str(seed),
                                "--batch-size", "128", "--buffer-tokens", "500", "--output-root", str(tmp_path / "sae")])["output_dir"]) / "sae.pt"


def _collect(tmp_path: Path, config_path: Path, sae: Path, *extra: str) -> Path:
    inputs = ["--input", str(tmp_path / "data" / "train.jsonl"), "--input", str(tmp_path / "data" / "heldout" / "congressional.jsonl")]
    return Path(collect_main(["--model", "bert", "--config", str(config_path), "--base", *inputs, "--sae", str(sae),
                              "--output-root", str(tmp_path / "act"), "--block-speeches", "16", *extra])["output_dir"])


def _reference(config, sae, text: str, layer: int) -> np.ndarray:
    model, tokenizer = load_model_and_tokenizer(config, config.model_name, torch.device("cpu"))
    model.eval()
    ids = tokenizer(text, add_special_tokens=False)["input_ids"]
    chunks, _ = chunk_token_ids(ids, config.max_length - 2, min_chunk_tokens=1)
    total, count = 0.0, 0
    for chunk in chunks:
        with torch.no_grad():
            state = model.base_model(input_ids=torch.tensor([[tokenizer.cls_token_id, *chunk, tokenizer.sep_token_id]]),
                                     output_hidden_states=True).hidden_states[layer][0, 1:-1]
            total = total + sae.encode(state).sum(0)
        count += len(chunk)
    return (total / count).numpy()


def test_sae_collection_equals_mean_of_token_features(tmp_path):
    config_path, act = _corpus(tmp_path, "bert")
    sae_path = _train(tmp_path, config_path, act)
    out = _collect(tmp_path, config_path, sae_path, "--tag", "sae_corpus", "--dtype", "float32", "--precision", "fp32")
    meta = json.loads((out / "meta.json").read_text())
    sae, _ = load_sae(sae_path, "cpu")
    assert meta["layers_stored"] == [2] and meta["representation"] == "token_sae"
    assert meta["hidden_size"] == sae.n_features and sorted(p.name for p in out.glob("layer_*.npy")) == ["layer_02.npy"]
    features = np.load(out / "layer_02.npy")
    config = parse_config(str(config_path))
    texts = [json.loads(line)["text"] for line in (tmp_path / "data" / "train.jsonl").read_text().splitlines()]
    for row in (0, 5, 17, 40):
        expected = _reference(config, sae, texts[row], 2)
        assert np.abs(features[row] - expected).max() <= 1e-4 * np.abs(expected).max() + 1e-5, row


def test_sae_collection_refuses_other_sae_and_resumes(tmp_path):
    config_path, act = _corpus(tmp_path, "bert")
    first, second = _train(tmp_path, config_path, act, seed=0), _train(tmp_path, config_path, act, seed=1)
    out = _collect(tmp_path, config_path, first, "--tag", "t", "--dtype", "float32")
    complete = np.load(out / "layer_02.npy").copy()
    with pytest.raises(RuntimeError, match="--overwrite"):
        _collect(tmp_path, config_path, second, "--tag", "t", "--dtype", "float32")
    (out / "progress.json").write_text(json.dumps({"completed_speeches": 16}))
    layer = np.load(out / "layer_02.npy", mmap_mode="r+")
    layer[16:] = 0
    layer.flush()
    _collect(tmp_path, config_path, first, "--tag", "t", "--dtype", "float32")
    assert np.array_equal(np.load(out / "layer_02.npy"), complete)


def test_verify_passes_sae_collection(tmp_path):
    config_path, act = _corpus(tmp_path, "bert")
    out = _collect(tmp_path, config_path, _train(tmp_path, config_path, act))
    assert out.name == "train+congressional_tokensae_layer02_k4_x2_seed0"
    report = verify_main([str(out), "--config", str(config_path), "--spot-check", "20", "--report", str(tmp_path / "r.json")])
    checks = report["collections"][str(out)]["checks"]
    assert report["ok"] is True, checks
    assert checks["row_identity"]["ok"] and checks["fidelity"]["worst_by_layer"].keys() == {2}
