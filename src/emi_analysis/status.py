"""Summarize training runs, during or after training, with what you need to decide whether to stop one.

    python -m emi_analysis.status                    # latest run in every folder under checkpoints/
    python -m emi_analysis.status checkpoints/val10k # latest runs under a folder
    python -m emi_analysis.status checkpoints/val10k/gpt2/run-job9822850
"""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime
from pathlib import Path

from .runinfo import format_duration

STALE_AFTER_SECONDS = 30 * 60


def _read_json(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _read_metrics(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def is_run_dir(path: Path) -> bool:
    return path.name.startswith("run-") and ((path / "metrics.csv").exists() or (path / "progress.json").exists()
                                             or (path / "run_summary.json").exists())


def find_latest_runs(root: Path) -> list[Path]:
    """Most recently modified run in each folder that holds runs (e.g. checkpoints/bert, checkpoints/val10k/gpt2)."""
    latest: dict[Path, Path] = {}
    for run_dir in (p for p in root.rglob("run-*") if p.is_dir() and is_run_dir(p)):
        current = latest.get(run_dir.parent)
        if current is None or run_dir.stat().st_mtime > current.stat().st_mtime:
            latest[run_dir.parent] = run_dir
    return sorted(latest.values())


def _with_change(value: str, base: str) -> str:
    if value == "":
        return "-"
    v, b = float(value), float(base)
    return f"{v:.4f}" if v == b or not b else f"{v:.4f} ({(v - b) / b:+.1%})"


def describe(run_dir: Path) -> list[str]:
    progress = _read_json(run_dir / "progress.json") or {}
    summary = _read_json(run_dir / "run_summary.json") or {}
    rows = _read_metrics(run_dir / "metrics.csv")
    model = progress.get("model") or summary.get("model") or (rows[0]["model"] if rows else "?")
    status = (summary.get("status") or progress.get("status") or "starting").upper()

    lines = ["=" * 100]
    updated = progress.get("updated_at")
    age = (datetime.now() - datetime.fromisoformat(updated)).total_seconds() if updated else None
    if status == "RUNNING" and age is not None and age > STALE_AFTER_SECONDS:
        status = f"RUNNING? no update for {format_duration(age)}; the job may have been killed (check squeue)"
    lines.append(f"{run_dir}  |  {model}  |  {status}")

    if progress:
        step, total = progress.get("step", 0), progress.get("total_steps", 0)
        parts = [
            f"step {step:,}/{total:,} ({progress.get('percent', 0):.1f}%)",
            f"epoch {progress.get('epoch', 0):.3f} of {progress.get('epochs_planned', 0):g}",
        ]
        if progress.get("steps_per_second"):
            parts.append(f"{progress['steps_per_second']:.2f} steps/s")
        parts.append(f"elapsed {format_duration(progress.get('elapsed_seconds'))}")
        if status == "RUNNING":
            parts.append(f"ETA {format_duration(progress.get('eta_seconds'))}")
        if updated:
            parts.append(f"updated {updated.replace('T', ' ')} ({format_duration(age)} ago)")
        lines.append("  " + " | ".join(parts))
        if progress.get("recent_train_loss") is not None:
            lines.append(f"  recent train loss {progress['recent_train_loss']:.4f} | learning rate {progress.get('learning_rate', 0):.2e}")
    if summary.get("error"):
        lines.append(f"  error: {summary['error']}")
    if summary.get("stop_reason"):
        lines.append(f"  stop reason: {summary['stop_reason']}")

    latest = progress.get("latest_checkpoint") or summary.get("latest_checkpoint") or ""
    lines.append(f"  latest checkpoint: {latest or 'none yet'}")

    if rows:
        metric = rows[0].get("loss_metric", "loss")
        base = rows[0]
        lines.append(f"  {metric} at each evaluation (change vs step 0 in brackets); train loss is since the previous evaluation:")
        lines.append(f"    {'step':>9}  {'epoch':>7}  {'event':<20} {'congressional':<22} {'general':<22} {'drift mean / max':<20} {'train':>8}")
        for row in rows:
            drift = f"{row['mean_seed_drift']} / {row['max_seed_drift']}" if row.get("mean_seed_drift") not in (None, "") else "-"
            train = f"{float(row['train_loss']):.4f}" if row.get("train_loss") else "-"
            lines.append(
                f"    {int(row['step']):>9,}  {float(row['epoch']):>7.3f}  {row.get('event', '') or '-':<20} "
                f"{_with_change(row['congressional_loss'], base['congressional_loss']):<22} "
                f"{_with_change(row['general_loss'], base['general_loss']):<22} {drift:<20} {train:>8}"
            )
        flagged = [r for r in rows if r.get("flags")]
        if flagged:
            lines.append("  flags:")
            lines.extend(f"    step {int(r['step']):,}: {r['flags']}" for r in flagged)
        else:
            lines.append("  flags: none")

    job = progress.get("slurm_job_id")
    if status.startswith("RUNNING") and job:
        lines.append(f"  to stop it: scancel {job}   (the run records a 'cancelled' summary; the latest checkpoint above is kept)")
    return lines


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Summarize training runs.")
    parser.add_argument("paths", nargs="*", default=["checkpoints"], help="Run folders, or folders to search for runs.")
    args = parser.parse_args(argv)

    run_dirs: list[Path] = []
    for raw in args.paths:
        path = Path(raw)
        run_dirs.extend([path] if is_run_dir(path) else find_latest_runs(path))
    if not run_dirs:
        print(f"No training runs found under {', '.join(args.paths)}")
        return
    for run_dir in run_dirs:
        print("\n".join(describe(run_dir)))


if __name__ == "__main__":
    main()
