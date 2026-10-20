from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from emi_analysis.clean_text_check import detokenize_wikitext, main as clean_main
from emi_analysis.train import main as train_main

from test_train import _make_run_inputs, _read_metrics


@pytest.mark.parametrize(
    ("raw", "clean"),
    [
        ("Robert Boulter is an English film , television and theatre actor . He had a role .",
         "Robert Boulter is an English film, television and theatre actor. He had a role."),
        ("the player 's contract ( signed in 1990 ) was worth $ 5 million ; it ended .",
         "the player's contract (signed in 1990) was worth $5 million; it ended."),
        ("He said \" we will win \" , but they did n't .", "He said \"we will win\", but they didn't."),
        ("1 @,@ 000 people paid 3 @.@ 5 % in a long @-@ term deal", "1,000 people paid 3.5% in a long-term deal"),
        ("the players ' union met : yes !", "the players' union met: yes!"),
        ("Already clean text, with (normal) spacing.", "Already clean text, with (normal) spacing."),
    ],
)
def test_detokenize_wikitext(raw, clean):
    assert detokenize_wikitext(raw) == clean


def _read_rows(path: Path) -> dict[tuple[int, str], dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return {(int(r["step"]), r["text_set"]): r for r in csv.DictReader(handle)}


def test_clean_text_check_reproduces_training_scores_and_rescores_clean_text(tmp_path):
    config_path = _make_run_inputs(tmp_path, "gpt2", seed_drift_threshold=2.0, general_loss_max_rel_increase=100.0)
    general = tmp_path / "data" / "heldout" / "general.jsonl"
    general.write_text("".join(
        json.dumps({"id": f"g{i}", "text": f"The committee , in its report ( filed in 199{i} ) , said the bill 's cost "
                                           f"was $ {i} million . It did n't pass , and the members ' vote failed ."}) + "\n"
        for i in range(6)), encoding="utf-8")
    run_dir = train_main(["--config", str(config_path), "--max-steps", "4", "--checkpoint-every", "2", "--run-name", "c"])

    summary = clean_main(["--model", "gpt2", "--config", str(config_path), "--run-dir", str(run_dir),
                          "--output-dir", str(tmp_path / "out")])

    rows = _read_rows(Path(summary["outputs"]["csv"]))
    metrics = _read_metrics(run_dir)
    # Raw WikiText is scored exactly as training scored it.
    assert float(rows[(0, "wikitext_raw")]["value"]) == pytest.approx(float(metrics[0]["general_loss"]), rel=1e-5)
    assert float(rows[(4, "wikitext_raw")]["value"]) == pytest.approx(float(metrics[-1]["general_loss"]), rel=1e-5)
    assert rows[(0, "wikitext_raw")]["matches_training_log"] == "True"
    # The detokenized copy is a different text, so GPT-2's byte-level tokenizer scores it differently.
    assert set(rows) == {(0, "wikitext_raw"), (4, "wikitext_raw"), (0, "wikitext_clean"), (4, "wikitext_clean")}
    assert rows[(0, "wikitext_clean")]["value"] != rows[(0, "wikitext_raw")]["value"]
    for text_set in ("wikitext_raw", "wikitext_clean"):
        assert isinstance(summary["rise"][text_set]["loss_change_pct"], float)
    assert "share_of_rise_from_formatting" in summary
