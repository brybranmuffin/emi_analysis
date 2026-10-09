from __future__ import annotations

import csv
from pathlib import Path
from typing import Iterable, Sequence


def append_metric_row(path: str | Path, row: dict[str, float | int | str]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["model", "step", "congressional_loss", "general_loss", "mean_seed_drift"]
    write_header = not path.exists() or path.stat().st_size == 0
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def read_metric_rows(path: str | Path) -> list[dict[str, str]]:
    path = Path(path)
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def validate_checkpoint(row: dict[str, float | int | str]) -> None:
    required = {"model", "step", "congressional_loss", "general_loss", "mean_seed_drift"}
    missing = required - set(row)
    if missing:
        raise ValueError(f"Metric row missing required keys: {sorted(missing)}")


def summarize_rows(rows: Sequence[dict[str, float | int | str]]) -> dict[str, float | int]:
    if not rows:
        return {"count": 0, "mean_general_loss": 0.0, "mean_seed_drift": 0.0}
    general_losses = [float(row["general_loss"]) for row in rows]
    seed_drifts = [float(row["mean_seed_drift"]) for row in rows]
    return {
        "count": len(rows),
        "mean_general_loss": sum(general_losses) / len(general_losses),
        "mean_seed_drift": sum(seed_drifts) / len(seed_drifts),
    }


def assert_checkpoint_conditions(model: str, row: dict[str, float | int | str]) -> None:
    """Fail fast when the verification signal indicates a problematic training run."""
    general_loss = float(row["general_loss"])
    drift = float(row["mean_seed_drift"])
    if general_loss > 2.0:
        raise RuntimeError(f"{model}: general-domain loss increased sharply; stop and inspect training.")
    if drift > 0.5:
        raise RuntimeError(f"{model}: mean seed drift is too large and still climbing; stop and report.")
