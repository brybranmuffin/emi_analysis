"""Verify collected activations on disk: complete, intact, correctly ordered, and equal to what the model produces.

    python -m emi_analysis.verify_activations activations/bert/run-job4757959/step350847/corpus [more dirs ...]

Per collection folder:
    status        meta.json says complete and progress.json covers every speech
    speech_ids    one unique ID per row, in exactly the order of the input files listed in meta.json
    files         every layer file present with shape [n_speeches, hidden] and the recorded dtype
    token_counts  every speech pooled at least one token; the total matches meta.json
    rows_written  full scan of every layer: no non-finite values, no all-zero row (unwritten rows are zeros)
    spot_check    a random sample of speeches recomputed from the checkpoint and compared with the stored rows
With several folders, also checks that all of them use the same speech order. Writes a JSON report and exits
non-zero if any check fails.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

from .runinfo import banner, format_duration, setup_logging

logger = logging.getLogger("emi_analysis")

SCAN_ROWS = 200_000
# Stored float16 rows vs a float32 recomputation: float16 keeps ~3 significant digits (relative error ~5e-4 of
# the speech's largest value); a wrong or misplaced row differs by far more.
SPOT_RELATIVE_TOLERANCE = 5e-3
SPOT_ABSOLUTE_TOLERANCE = 1e-3


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
            empty = ~block.any(axis=1) & (token_counts[start : start + SCAN_ROWS] > 0)
            zeros += int(empty.sum())
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


def spot_check(out: Path, meta: dict, config_path: str, sample: int, seed: int, require_gpu: bool) -> dict:
    from .collect_activations import main as collect_main

    n = meta["n_speeches"]
    rng = np.random.default_rng(seed)
    picks = sorted(rng.choice(n, size=min(sample, n), replace=False).tolist())
    rows = read_input_rows(meta["inputs"], set(picks))
    checkpoint = meta["checkpoint"]
    with tempfile.TemporaryDirectory() as tmp:
        sample_path = Path(tmp) / "sample.jsonl"
        sample_path.write_text("".join(json.dumps(rows[i]) + "\n" for i in picks), encoding="utf-8")
        argv = ["--model", meta["model"], "--config", config_path, "--input", str(sample_path), "--tag", "spot",
                "--output-root", str(Path(tmp) / "act"), "--dtype", "float32"]
        argv += ["--base"] if checkpoint.get("run") == "base" else ["--checkpoint", checkpoint["path"]]
        if require_gpu:
            argv.append("--require-gpu")
        fresh_dir = Path(collect_main(argv)["output_dir"])
        worst, worst_row = 0.0, None
        for layer in range(meta["n_layers"]):
            stored = np.load(out / f"layer_{layer:02d}.npy", mmap_mode="r")
            fresh = np.load(fresh_dir / f"layer_{layer:02d}.npy")
            for k, row in enumerate(picks):
                expected = fresh[k].astype(np.float64)
                error = float(np.abs(stored[row].astype(np.float64) - expected).max())
                allowed = SPOT_RELATIVE_TOLERANCE * float(np.abs(expected).max()) + SPOT_ABSOLUTE_TOLERANCE
                if error / allowed > worst:
                    worst, worst_row = error / allowed, (layer, row, error)
    ok = worst <= 1.0
    detail = (f"{len(picks)} speeches recomputed from {checkpoint['path']}: all layers match "
              f"(worst error {worst:.2f} of tolerance)" if ok else
              f"mismatch: layer {worst_row[0]}, row {worst_row[1]} differs by {worst_row[2]:.4g} ({worst:.1f}x tolerance)")
    return _check(ok, detail, speeches=len(picks), worst_fraction_of_tolerance=round(worst, 4))


def verify(out: Path, args) -> dict:
    started = time.time()
    meta = json.loads((out / "meta.json").read_text(encoding="utf-8"))
    progress = json.loads((out / "progress.json").read_text(encoding="utf-8")) if (out / "progress.json").exists() else {}
    n = meta["n_speeches"]
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
    zero_tokens = int((token_counts == 0).sum())
    total = int(token_counts.sum())
    checks["token_counts"] = _check(token_counts.shape == (n,) and zero_tokens == 0 and total == meta.get("tokens_pooled"),
                                    f"{zero_tokens:,} speeches with no tokens; {total:,} tokens pooled "
                                    f"(meta.json: {meta.get('tokens_pooled', 'n/a'):,})")
    checks["files"], checks["rows_written"] = scan_layers(out, meta, token_counts)
    if args.spot_check > 0 and checks["files"]["ok"]:
        config_path = args.config or f"configs/{meta['model']}_continued_pretraining.json"
        checks["spot_check"] = spot_check(out, meta, config_path, args.spot_check, args.seed, args.require_gpu)
        setup_logging(args.report_path.with_suffix(".log"), mode="a")  # the recomputation replaced the log handlers
    ok = all(c["ok"] for c in checks.values())
    banner(logger, f"{'PASSED' if ok else 'FAILED'}: {out}", {
        **{name: f"{'ok  ' if c['ok'] else 'FAIL'} {c['detail']}" for name, c in checks.items()},
        "elapsed": format_duration(time.time() - started),
    })
    return {"ok": ok, "checks": checks, "n_speeches": n, "model": meta["model"], "checkpoint": meta["checkpoint"]["path"]}


def build_cli() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Verify collected activations.")
    parser.add_argument("dirs", nargs="+", help="Collection folders (each holding meta.json and layer_*.npy).")
    parser.add_argument("--config", default=None, help="Training config for the spot-check (default: configs/<model>_continued_pretraining.json).")
    parser.add_argument("--spot-check", type=int, default=32, help="Speeches to recompute per collection (0 to skip).")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--report", default="activations/verify_report.json")
    parser.add_argument("--require-gpu", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> dict:
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
