"""Verify collected activations on disk: complete, intact, correctly ordered, and faithful to the model.

    python -m emi_analysis.verify_activations activations/bert/run-job4757959/step350847/corpus [more dirs ...]

Per collection folder:
    status        meta.json says complete and progress.json covers every speech
    speech_ids    one unique ID per row, in exactly the order of the input files listed in meta.json
    files         every layer file present with shape [n_speeches, hidden] and the recorded dtype
    token_counts  each speech's pooled-token count equals a fresh tokenization's count (special IDs excluded)
    rows_written  full scan of every layer: no non-finite values, no all-zero row (unwritten rows are zeros)
    row_identity  for a random sample, layer 0 (the embedding output, precise in any format) and the token count
                  match a fresh computation of that same speech, so no row holds another speech
    fidelity      for the same sample, every layer agrees with a strict-float32 recomputation after centering
                  (worst speech's cosine >= 0.99), so the stored precision preserves the speech-specific signal
With several folders, also checks that all of them use the same speech order. Writes a JSON report and exits
non-zero if any check fails.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

from .runinfo import banner, format_duration, setup_logging

logger = logging.getLogger("emi_analysis")

SCAN_ROWS = 200_000
# Layer 0 stored as float16 vs a float32 recomputation: float16 keeps ~3 significant digits; another speech's row
# differs by far more.
IDENTITY_RELATIVE_TOLERANCE = 5e-3
IDENTITY_ABSOLUTE_TOLERANCE = 1e-3
# Centered cosine between stored and float32 vectors; bf16 ModernBERT layers 16-22 fall to 0.10-0.98.
FIDELITY_MIN = 0.99


def _check(ok: bool, detail: str, **extra) -> dict:
    return {"ok": bool(ok), "detail": detail, **extra}


def read_input_ids(paths: list[str]) -> list[str]:
    ids = []
    for path in paths:
        with open(path, encoding="utf-8") as handle:
            ids.extend(str(json.loads(line)["speech_id"]) for line in handle if line.strip())
    return ids


def read_input_rows(paths: list[str], wanted: set[int]) -> dict[int, dict]:
    rows, index = {}, 0
    for path in paths:
        with open(path, encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                if index in wanted:
                    rows[index] = json.loads(line)
                index += 1
    return rows


def expected_token_counts(meta: dict, config, num_proc: int) -> np.ndarray:
    """Tokens each speech should have pooled: its chunk tokens minus any special IDs (wrappers, in-text [UNK] ...)."""
    import datasets
    from transformers import AutoTokenizer

    from .collect_activations import tokenize_speeches
    from .train import load_jsonl_dataset

    path = meta["checkpoint"]["path"]
    tokenizer = AutoTokenizer.from_pretrained(path, cache_dir=config.local_cache_dir)
    window = meta["chunking"]["window"]
    speeches = datasets.concatenate_datasets([load_jsonl_dataset(p, config.datasets_cache_dir) for p in meta["inputs"]])
    chunks = tokenize_speeches(speeches, tokenizer, window, num_proc)
    special = np.asarray(sorted(set(tokenizer.all_special_ids)))

    def count(batch):
        return {"pooled": [int((~np.isin(np.asarray(ids), special)).sum()) for ids in batch["input_ids"]]}

    counted = chunks.map(count, batched=True, num_proc=num_proc if num_proc > 1 else None, remove_columns=["input_ids", "length"])
    columns = counted.with_format("numpy")[:]
    return np.bincount(columns["speech"], weights=columns["pooled"], minlength=meta["n_speeches"]).astype(np.int64)


def scan_layers(out: Path, meta: dict, token_counts: np.ndarray) -> tuple[dict, dict]:
    n, hidden, dtype = meta["n_speeches"], meta["hidden_size"], np.dtype(meta["dtype"])
    missing_files, bad_shape = [], []
    zero_rows, nonfinite, max_abs = {}, {}, {}
    for layer in range(meta["n_layers"]):
        path = out / f"layer_{layer:02d}.npy"
        if not path.exists():
            missing_files.append(layer)
            continue
        array = np.load(path, mmap_mode="r")
        if array.shape != (n, hidden) or array.dtype != dtype:
            bad_shape.append(f"layer {layer}: {array.shape} {array.dtype}")
            continue
        zeros, bad, peak = 0, 0, 0.0
        for start in range(0, n, SCAN_ROWS):
            block = np.asarray(array[start : start + SCAN_ROWS], dtype=np.float32)
            finite = np.isfinite(block)
            bad += int((~finite).sum())
            zeros += int((~block.any(axis=1) & (token_counts[start : start + SCAN_ROWS] > 0)).sum())
            peak = max(peak, float(np.abs(np.where(finite, block, 0)).max(initial=0.0)))
        if zeros:
            zero_rows[layer] = zeros
        if bad:
            nonfinite[layer] = bad
        max_abs[layer] = round(peak, 3)
    files = _check(not missing_files and not bad_shape,
                   "all present with the recorded shape and dtype" if not missing_files and not bad_shape
                   else f"missing layers {missing_files}; wrong shape/dtype: {bad_shape}")
    problems = [f"layer {k}: {v:,} all-zero rows" for k, v in zero_rows.items()] + \
               [f"layer {k}: {v:,} non-finite values" for k, v in nonfinite.items()]
    rows = _check(not problems, "; ".join(problems) or "every row of every layer written and finite", max_abs_by_layer=max_abs)
    return files, rows


def recompute_sample(meta: dict, config_path: str, picks: list[int], tmp: Path, require_gpu: bool) -> Path:
    """Recompute the sampled speeches with the same checkpoint in strict float32."""
    from .collect_activations import main as collect_main

    rows = read_input_rows(meta["inputs"], set(picks))
    sample_path = tmp / "sample.jsonl"
    sample_path.write_text("".join(json.dumps(rows[i]) + "\n" for i in picks), encoding="utf-8")
    checkpoint = meta["checkpoint"]
    argv = ["--model", meta["model"], "--config", config_path, "--input", str(sample_path), "--tag", "spot",
            "--output-root", str(tmp / "act"), "--dtype", "float32", "--precision", "fp32"]
    argv += ["--base"] if checkpoint.get("run") == "base" else ["--checkpoint", checkpoint["path"]]
    if require_gpu:
        argv.append("--require-gpu")
    return Path(collect_main(argv)["output_dir"])


def identity_and_fidelity(out: Path, meta: dict, picks: list[int], fresh: Path, token_counts: np.ndarray) -> tuple[dict, dict]:
    fresh_counts = np.load(fresh / "token_counts.npy")
    stored0 = np.load(out / "layer_00.npy", mmap_mode="r")
    fresh0 = np.load(fresh / "layer_00.npy")
    wrong = []
    for k, row in enumerate(picks):
        expected = fresh0[k].astype(np.float64)
        error = float(np.abs(stored0[row].astype(np.float64) - expected).max())
        allowed = IDENTITY_RELATIVE_TOLERANCE * float(np.abs(expected).max()) + IDENTITY_ABSOLUTE_TOLERANCE
        if error > allowed or int(token_counts[row]) != int(fresh_counts[k]):
            wrong.append(row)
    identity = _check(not wrong, f"{len(picks)} sampled rows hold their own speech (layer 0 and token count match)" if not wrong
                      else f"{len(wrong)} sampled rows do not match their speech: rows {wrong[:10]}", speeches=len(picks))

    worst_by_layer = {}
    for layer in range(meta["n_layers"]):
        stored = np.asarray(np.load(out / f"layer_{layer:02d}.npy", mmap_mode="r")[picks], dtype=np.float64)
        reference = np.load(fresh / f"layer_{layer:02d}.npy").astype(np.float64)
        centre = reference.mean(axis=0)
        a, b = stored - centre, reference - centre
        norms = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1)
        cos = np.where(norms > 0, (a * b).sum(axis=1) / np.where(norms > 0, norms, 1), 1.0)
        worst_by_layer[layer] = round(float(cos.min()), 4)
    failing = [layer for layer, value in worst_by_layer.items() if value < FIDELITY_MIN]
    worst = min(worst_by_layer.values())
    fidelity = _check(not failing,
                      f"all layers match float32 (worst centered cosine {worst:.4f})" if not failing else
                      f"layers {failing} lose the speech-specific signal (centered cosine "
                      f"{', '.join(f'{worst_by_layer[l]:.2f}' for l in failing)} < {FIDELITY_MIN}); re-collect with --precision fp32",
                      failing_layers=failing, worst_centered_cosine=worst, worst_by_layer=worst_by_layer)
    return identity, fidelity


def verify(out: Path, args) -> dict:
    from .train import parse_config

    started = time.time()
    meta = json.loads((out / "meta.json").read_text(encoding="utf-8"))
    progress = json.loads((out / "progress.json").read_text(encoding="utf-8")) if (out / "progress.json").exists() else {}
    n = meta["n_speeches"]
    config_path = args.config or f"configs/{meta['model']}_continued_pretraining.json"
    config = parse_config(config_path)
    num_proc = args.num_proc or int(os.environ.get("SLURM_CPUS_PER_TASK", "1"))
    checks = {}
    checks["status"] = _check(meta.get("status") == "complete" and progress.get("completed_speeches") == n,
                              f"meta status {meta.get('status')!r}, progress {progress.get('completed_speeches')}/{n:,}")

    stored_ids = (out / "speech_ids.txt").read_text(encoding="utf-8").split()
    expected_ids = read_input_ids(meta["inputs"])
    first_diff = next((i for i, (a, b) in enumerate(zip(stored_ids, expected_ids)) if a != b), None)
    ids_ok = len(stored_ids) == n == len(expected_ids) and first_diff is None and len(set(stored_ids)) == n
    checks["speech_ids"] = _check(ids_ok, f"{len(stored_ids):,} IDs, unique and in input order" if ids_ok else
                                  f"{len(stored_ids):,} stored vs {len(expected_ids):,} in inputs; first difference at row {first_diff}")

    token_counts = np.load(out / "token_counts.npy")
    expected = expected_token_counts(meta, config, num_proc)
    mismatched = int((token_counts != expected).sum()) if token_counts.shape == expected.shape else n
    in_text = int(meta.get("tokens_before_special_exclusion", meta.get("tokens_pooled", 0))) - int(token_counts.sum(dtype=np.int64))
    checks["token_counts"] = _check(
        mismatched == 0 and int((token_counts == 0).sum()) == 0,
        f"{n - mismatched:,} of {n:,} speeches pooled exactly the expected tokens; {int((token_counts == 0).sum()):,} with none"
        + (f"; {in_text:,} in-text special tokens (e.g. [UNK]) excluded by design" if in_text > 0 else "")
        if mismatched == 0 else f"{mismatched:,} speech{'es' if mismatched != 1 else ''} pooled a different number of tokens than expected",
        mismatched_speeches=mismatched)

    checks["files"], checks["rows_written"] = scan_layers(out, meta, token_counts)
    if args.spot_check > 0 and checks["files"]["ok"]:
        picks = sorted(np.random.default_rng(args.seed).choice(n, size=min(args.spot_check, n), replace=False).tolist())
        with tempfile.TemporaryDirectory() as tmp:
            fresh = recompute_sample(meta, config_path, picks, Path(tmp), args.require_gpu)
            setup_logging(args.report_path.with_suffix(".log"), mode="a")  # the recomputation replaced the log handlers
            checks["row_identity"], checks["fidelity"] = identity_and_fidelity(out, meta, picks, fresh, token_counts)
    ok = all(c["ok"] for c in checks.values())
    banner(logger, f"{'PASSED' if ok else 'FAILED'}: {out}", {
        **{name: f"{'ok  ' if c['ok'] else 'FAIL'} {c['detail']}" for name, c in checks.items()},
        "precision of stored vectors": meta.get("precision", "bf16 (collected before --precision existed)"),
        "elapsed": format_duration(time.time() - started),
    })
    return {"ok": ok, "checks": checks, "n_speeches": n, "model": meta["model"], "checkpoint": meta["checkpoint"]["path"]}


def build_cli() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Verify collected activations.")
    parser.add_argument("dirs", nargs="+", help="Collection folders (each holding meta.json and layer_*.npy).")
    parser.add_argument("--config", default=None, help="Training config (default: configs/<model>_continued_pretraining.json).")
    parser.add_argument("--spot-check", type=int, default=64, help="Speeches to recompute per collection (0 to skip).")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-proc", type=int, default=None)
    parser.add_argument("--report", default="activations/verify_report.json")
    parser.add_argument("--require-gpu", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> dict:
    import datasets

    datasets.disable_progress_bars()
    args = build_cli().parse_args(argv)
    args.report_path = Path(args.report)
    args.report_path.parent.mkdir(parents=True, exist_ok=True)
    setup_logging(args.report_path.with_suffix(".log"), mode="a")
    results = {d: verify(Path(d), args) for d in args.dirs}
    aligned = None
    if len(args.dirs) > 1:
        orders = {d: (Path(d) / "speech_ids.txt").read_text(encoding="utf-8") for d in args.dirs}
        aligned = len(set(orders.values())) == 1
        logger.info(f"Speech order identical across all {len(args.dirs)} collections: {aligned}")
    report = {"ok": all(r["ok"] for r in results.values()) and aligned is not False, "aligned_across_collections": aligned,
              "collections": results}
    args.report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    logger.info(f"Overall: {'PASSED' if report['ok'] else 'FAILED'} (report: {args.report_path})")
    return report


if __name__ == "__main__":
    sys.exit(0 if main()["ok"] else 1)
