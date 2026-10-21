"""Seed-axis check: did adaptation move the EMI axis, or just shift every embedding?

Training's seed-drift metric measures how far each seed word's embedding moved. Continued pretraining often shifts
all embeddings in a shared direction, which inflates that number without changing the geometry EMI relies on. This
script re-embeds the evidence and intuition seed words at step 0 and at every checkpoint (exactly as training does:
mean-pooled last hidden state of each word on its own) and measures what EMI actually uses:

    axis_cosine          cosine between the (evidence pole - intuition pole) direction now and at step 0 (1 = same)
    relational_spearman  rank correlation of all seed-word pairwise cosine similarities now vs step 0 (1 = same pattern)
    axis_order_spearman  rank correlation of each seed word's position along the axis now vs step 0
    separation_accuracy  share of seed words closer to their own (leave-one-out) pole than to the other pole

It then joins these with congressional loss (training log) and clean general-domain loss (clean_text_check output)
into one per-checkpoint adaptation table.

    python -m emi_analysis.seed_axis_check --model qwen3            # latest completed run, every checkpoint

Writes <output-dir>/<model>_<run>.csv and .json; the log is appended to <output-dir>/<model>_check.log.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import time
from datetime import datetime
from pathlib import Path

import torch
import torch.nn.functional as F
from scipy.stats import spearmanr

from .clean_text_check import latest_completed_run, read_metrics, resolve_checkpoint
from .runinfo import banner, environment_info, format_duration, setup_logging
from .train import embed_words, load_lexicon, load_model_and_tokenizer, parse_config, seed_drift

logger = logging.getLogger("emi_analysis")

# Provisional reading guide (not calibrated; the handoff gives no numbers).
STABLE_MIN = 0.90       # axis cosine and relational correlation at or above this: axis stable
MOVED_BELOW = 0.70      # either below this: axis moved
SEPARATION_TOLERANCE = 0.05
SEPARATION_MOVED = 0.10


def _cos(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return F.cosine_similarity(a, b, dim=-1)


def _axis_scores(ev: torch.Tensor, intu: torch.Tensor) -> torch.Tensor:
    """Position of every seed word along the axis: cos(word, evidence pole) - cos(word, intuition pole)."""
    words = torch.cat([ev, intu])
    return _cos(words, ev.mean(0)) - _cos(words, intu.mean(0))


def _separation(ev: torch.Tensor, intu: torch.Tensor) -> tuple[float, float]:
    """Share of words closer to their own leave-one-out pole than to the other pole, and the mean margin."""
    margins = []
    for own, other in ((ev, intu), (intu, ev)):
        loo_poles = (own.sum(0, keepdim=True) - own) / (len(own) - 1)
        margins.append(_cos(own, loo_poles) - _cos(own, other.mean(0)))
    margins = torch.cat(margins)
    return float((margins > 0).double().mean()), float(margins.mean())


def seed_axis_metrics(ev0: torch.Tensor, int0: torch.Tensor, ev: torch.Tensor, intu: torch.Tensor) -> dict[str, float]:
    """Compare seed embeddings now (``ev``, ``intu``) with step 0 (``ev0``, ``int0``)."""
    mean_drift, max_drift = seed_drift(torch.cat([ev0, int0]), torch.cat([ev, intu]))
    axis0, axis = ev0.mean(0) - int0.mean(0), ev.mean(0) - intu.mean(0)
    words0, words = F.normalize(torch.cat([ev0, int0]), dim=-1), F.normalize(torch.cat([ev, intu]), dim=-1)
    upper = torch.triu_indices(len(words), len(words), offset=1)
    sims0, sims = (words0 @ words0.T)[upper[0], upper[1]], (words @ words.T)[upper[0], upper[1]]
    relational, _ = spearmanr(sims0.numpy(), sims.numpy())
    order, _ = spearmanr(_axis_scores(ev0, int0).numpy(), _axis_scores(ev, intu).numpy())
    accuracy, margin = _separation(ev, intu)
    accuracy0, _ = _separation(ev0, int0)
    return {
        "mean_drift": mean_drift,
        "max_drift": max_drift,
        "axis_cosine": float(_cos(axis0, axis)),
        "relational_spearman": float(relational),
        "axis_order_spearman": float(order),
        "separation_accuracy": accuracy,
        "separation_margin": margin,
        "separation_accuracy_step0": accuracy0,
    }


def verdict(m: dict[str, float]) -> str:
    drop = m["separation_accuracy_step0"] - m["separation_accuracy"]
    if m["axis_cosine"] < MOVED_BELOW or m["relational_spearman"] < MOVED_BELOW or drop > SEPARATION_MOVED:
        return "moved"
    if m["axis_cosine"] >= STABLE_MIN and m["relational_spearman"] >= STABLE_MIN and drop <= SEPARATION_TOLERANCE:
        return "stable"
    return "watch"


def build_cli() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Axis-level seed drift per checkpoint, joined into an adaptation table.")
    parser.add_argument("--model", required=True, choices=["bert", "gpt2", "modernbert", "qwen3"])
    parser.add_argument("--config", default=None, help="Training config (default: configs/<model>_continued_pretraining.json).")
    parser.add_argument("--run-dir", default=None, help="Training run folder (default: latest completed run in the config's output_dir).")
    parser.add_argument("--clean-text-dir", default="results/clean_text_check", help="Where clean_text_check wrote its CSVs.")
    parser.add_argument("--output-dir", default="results/seed_axis_check")
    parser.add_argument("--require-gpu", action="store_true")
    return parser


def _loss(value: float, metric: str) -> float:
    return math.log(value) if metric == "perplexity" else value


def _pct(new: float, old: float) -> float | str:
    return round(100 * (new - old) / old, 2) if old else ""


def run_check(args: argparse.Namespace) -> dict:
    started = time.time()
    config = parse_config(args.config or f"configs/{args.model}_continued_pretraining.json")
    run_dir = Path(args.run_dir) if args.run_dir else latest_completed_run(Path(config.output_dir))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.require_gpu and device.type != "cuda":
        raise RuntimeError("--require-gpu was set but CUDA is not available on this node.")

    evidence, intuition = load_lexicon(config.evidence_lexicon_path), load_lexicon(config.intuition_lexicon_path)
    seed_words = evidence + intuition  # same order and single batch as training, so drift reproduces exactly
    metrics = {int(r["step"]): r for r in read_metrics(run_dir)}
    steps = sorted(metrics)
    clean_path = Path(args.clean_text_dir) / f"{args.model}_{run_dir.name}.csv"
    clean = {}
    if clean_path.exists():
        with clean_path.open(newline="", encoding="utf-8") as handle:
            clean = {int(r["step"]): r for r in csv.DictReader(handle) if r["text_set"] == "wikitext_clean"}

    banner(logger, f"SEED-AXIS CHECK: {args.model}", {
        "training run": run_dir,
        "seed words": f"{len(evidence)} evidence + {len(intuition)} intuition",
        "checkpoints": f"{len(steps)} (step 0 and every saved checkpoint)",
        "clean general-domain scores": clean_path if clean else f"not found ({clean_path}); run clean_text_check --all-checkpoints first",
        "reading guide (provisional)": f"stable: axis cosine and relational >= {STABLE_MIN}, separation within "
                                       f"{SEPARATION_TOLERANCE:.0%} of step 0; moved: either < {MOVED_BELOW} or separation "
                                       f"down > {SEPARATION_MOVED:.0%}",
        **environment_info(),
    })

    embeddings: dict[int, torch.Tensor] = {}
    for step in steps:
        name_or_path = config.model_name if step == 0 else str(resolve_checkpoint(metrics[step]["checkpoint"], run_dir))
        model, tokenizer = load_model_and_tokenizer(config, name_or_path, device)
        embeddings[step] = embed_words(model, tokenizer, seed_words, device)
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    n_ev = len(evidence)
    base = embeddings[0]
    metric = metrics[0]["loss_metric"]
    cong0 = _loss(float(metrics[0]["congressional_loss"]), metric)
    gen0 = float(clean[0]["loss"]) if 0 in clean else None
    rows = []
    for step in steps:
        now = embeddings[step]
        m = seed_axis_metrics(base[:n_ev], base[n_ev:], now[:n_ev], now[n_ev:])
        logged = metrics[step].get("mean_seed_drift", "")
        row = {
            "model": args.model, "step": step, "epoch": metrics[step]["epoch"], "event": metrics[step].get("event", ""),
            "checkpoint": metrics[step]["checkpoint"], "loss_metric": metric,
            "congressional": float(metrics[step]["congressional_loss"]),
            "congressional_loss_change_pct": _pct(_loss(float(metrics[step]["congressional_loss"]), metric), cong0),
            "general_clean": round(float(clean[step]["value"]), 6) if step in clean else "",
            "general_clean_loss_change_pct": _pct(float(clean[step]["loss"]), gen0) if step in clean and gen0 else "",
            **{k: round(v, 6) for k, v in m.items()},
            "training_log_drift": logged,
            "matches_training_log": abs(m["mean_drift"] - float(logged)) <= 1e-3 if logged not in ("", None) else "",
            "verdict": "baseline" if step == 0 else verdict(m),
        }
        rows.append(row)
        logger.info(
            f"step {step:>9,} | congressional {row['congressional_loss_change_pct']!s:>7}% | general(clean) "
            f"{row['general_clean_loss_change_pct']!s:>7}% | drift {m['mean_drift']:.3f} | axis cos {m['axis_cosine']:.3f} "
            f"| relational {m['relational_spearman']:.3f} | order {m['axis_order_spearman']:.3f} "
            f"| separation {m['separation_accuracy']:.2f} (step 0 {m['separation_accuracy_step0']:.2f}) | {row['verdict']}"
        )

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    csv_path, json_path = out / f"{args.model}_{run_dir.name}.csv", out / f"{args.model}_{run_dir.name}.json"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    final = rows[-1]
    summary = {
        "model": args.model, "run_dir": str(run_dir), "final": final,
        "verdicts": {r["step"]: r["verdict"] for r in rows},
        "drift_reproduces_training_log": all(r["matches_training_log"] for r in rows if r["matches_training_log"] != ""),
        "clean_text_scores_included": bool(clean),
        "outputs": {"csv": str(csv_path), "json": str(json_path), "log": str(out / f"{args.model}_check.log")},
        "finished_at": datetime.now().isoformat(timespec="seconds"), "elapsed_seconds": round(time.time() - started, 1),
    }
    json_path.write_text(json.dumps(summary, indent=2, default=str) + "\n", encoding="utf-8")
    banner(logger, f"SEED-AXIS CHECK RESULT: {args.model} (final step {final['step']:,})", {
        "raw drift (training metric)": f"{final['mean_drift']:.3f} mean / {final['max_drift']:.3f} max",
        "axis cosine": f"{final['axis_cosine']:.3f}",
        "relational stability": f"{final['relational_spearman']:.3f}",
        "seed order along axis": f"{final['axis_order_spearman']:.3f}",
        "pole separation": f"{final['separation_accuracy']:.2f} (step 0: {final['separation_accuracy_step0']:.2f})",
        "verdict": final["verdict"],
        "drift reproduces training log": summary["drift_reproduces_training_log"],
        "results": f"{csv_path}, {json_path}",
        "elapsed": format_duration(time.time() - started),
    })
    if not summary["drift_reproduces_training_log"]:
        logger.warning("Recomputed raw drift differs from the training log; check the config, lexicons and run folder.")
    return summary


def main(argv: list[str] | None = None) -> dict:
    args = build_cli().parse_args(argv)
    setup_logging(Path(args.output_dir) / f"{args.model}_check.log", mode="a")
    try:
        return run_check(args)
    except Exception:
        logger.exception("Seed-axis check failed")
        raise


if __name__ == "__main__":
    main()
