from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import pytest
import torch

from emi_analysis.keyword_activations import main as keyword_main
from emi_analysis.keyword_occurrences import main as occurrences_main
from emi_analysis.train import load_model_and_tokenizer, parse_config

from test_train import WORDS, _make_run_inputs, _write_jsonl


def _inputs(tmp_path: Path, family: str, tiny_qwen3_path: str, rows: list[dict] | None = None) -> Path:
    overrides = {"max_length": 16}
    if family == "qwen3":
        overrides["model_name"] = tiny_qwen3_path
    config_path = _make_run_inputs(tmp_path, family, **overrides)
    if rows is None:
        rng = random.Random(1)
        rows = [{"speech_id": f"s{i}", "congress": 46 + 9 * (i % 8),
                 "text": " ".join(rng.choice(WORDS) for _ in range(rng.randint(5, 40))) + " the evidence"} for i in range(40)]
    _write_jsonl(tmp_path / "data" / "train.jsonl", rows)
    return config_path


def _collect(tmp_path: Path, config_path: Path, family: str) -> tuple[Path, Path]:
    occ = Path(occurrences_main(["--config", str(config_path), "--per-era", "3", "--unbalanced", "4", "--num-proc", "1",
                                 "--output-root", str(tmp_path / "occ")])["output_dir"])
    out = Path(keyword_main(["--model", family, "--config", str(config_path), "--base", "--occurrences", str(occ),
                             "--precision", "fp32", "--output-root", str(tmp_path / "kw"), "--batch-chunks", "5"])["output_dir"])
    return occ, out


def _states(model, ids: list[int]) -> tuple[torch.Tensor, ...]:
    with torch.no_grad():
        return model.base_model(input_ids=torch.tensor([ids]), output_hidden_states=True).hidden_states


@pytest.mark.parametrize("family", ["bert", "gpt2", "qwen3"])
def test_occurrence_state_equals_token_state_in_its_chunk(tmp_path, family, tiny_qwen3_path):
    config_path = _inputs(tmp_path, family, tiny_qwen3_path)
    occ_dir, out = _collect(tmp_path, config_path, family)
    config = parse_config(str(config_path))
    model, tokenizer = load_model_and_tokenizer(config, config.model_name, torch.device("cpu"))
    model.eval()
    occ = np.load(occ_dir / "occurrences.npz")
    words = json.loads((occ_dir / "keywords.json").read_text())["words"]
    texts = [json.loads(line)["text"] for line in (tmp_path / "data" / "train.jsonl").read_text().splitlines()]
    meta = json.loads((out / "meta.json").read_text())
    wrap = tokenizer.cls_token_id is not None and tokenizer.sep_token_id is not None
    window = config.max_length - (2 if wrap else 0)
    layers = [np.load(out / f"occ_layer_{i:02d}.npy") for i in range(meta["n_layers"])]
    assert layers[0].shape == (len(occ["row"]), model.config.hidden_size)

    for i in range(0, len(occ["row"]), 7):
        text, start, end = texts[occ["row"][i]], int(occ["start"][i]), int(occ["end"][i])
        enc = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
        pieces = [t for t, (a, b) in enumerate(enc["offset_mapping"]) if a < end and b > start]
        assert tokenizer.decode([enc["input_ids"][t] for t in pieces]).strip().lower() == words[occ["keyword"][i]]
        chunk = pieces[0] // window
        ids = enc["input_ids"][chunk * window : (chunk + 1) * window]
        ids = [tokenizer.cls_token_id, *ids, tokenizer.sep_token_id] if wrap else ids
        positions = [t - chunk * window + (1 if wrap else 0) for t in pieces if t // window == chunk]
        states = _states(model, ids)
        for layer, state in enumerate(states):
            expected = state[0, positions].mean(0).numpy()
            assert np.abs(layers[layer][i] - expected).max() <= 1e-4 * np.abs(expected).max() + 1e-5, (i, layer)


def test_alone_vectors_exclude_special_tokens(tmp_path, tiny_qwen3_path):
    config_path = _inputs(tmp_path, "bert", tiny_qwen3_path)
    occ_dir, out = _collect(tmp_path, config_path, "bert")
    config = parse_config(str(config_path))
    model, tokenizer = load_model_and_tokenizer(config, config.model_name, torch.device("cpu"))
    model.eval()
    words = json.loads((occ_dir / "keywords.json").read_text())["words"]
    alone = np.load(out / "alone_layer_02.npy")
    assert alone.shape[0] == len(words)
    word = words.index("evidence")
    ids = tokenizer("evidence", add_special_tokens=False)["input_ids"]
    states = _states(model, [tokenizer.cls_token_id, *ids, tokenizer.sep_token_id])
    expected = states[2][0, 1 : 1 + len(ids)].mean(0).numpy()
    assert np.abs(alone[word] - expected).max() <= 1e-4 * np.abs(expected).max() + 1e-5


def test_straddling_span_uses_first_tokens_chunk(tmp_path, tiny_qwen3_path):
    config_path = _make_run_inputs(tmp_path, "bert", max_length=16)
    config = parse_config(str(config_path))
    _, tokenizer = load_model_and_tokenizer(config, config.model_name, torch.device("cpu"))
    window = 14
    keyword = next(w for w in ("evidence", "believe", "study", "data", "feel", "heart")
                   if len(tokenizer(w, add_special_tokens=False)["input_ids"]) >= 2)
    filler = next(w for w in ("the", "a", "i", "x", ",", ".") if len(tokenizer(w, add_special_tokens=False)["input_ids"]) == 1)
    text = " ".join([filler] * (window - 1) + [keyword] + ["vote"] * 5)  # keyword's first piece is the window's last token
    _write_jsonl(tmp_path / "data" / "train.jsonl", [{"speech_id": "s0", "congress": 50, "text": text}])
    occ_dir, out = _collect(tmp_path, config_path, "bert")
    meta = json.loads((out / "meta.json").read_text())
    assert meta["straddling_spans"] == 1

    model, _ = load_model_and_tokenizer(config, config.model_name, torch.device("cpu"))
    model.eval()
    ids = tokenizer(text, add_special_tokens=False)["input_ids"][:window]
    states = _states(model, [tokenizer.cls_token_id, *ids, tokenizer.sep_token_id])
    occ = np.load(occ_dir / "occurrences.npz")
    row = int(np.flatnonzero(np.asarray(json.loads((occ_dir / "keywords.json").read_text())["words"])[occ["keyword"]] == keyword)[0])
    expected = states[1][0, window].numpy()  # only the piece inside the first chunk
    assert np.abs(np.load(out / "occ_layer_01.npy")[row] - expected).max() <= 1e-4 * np.abs(expected).max() + 1e-5
