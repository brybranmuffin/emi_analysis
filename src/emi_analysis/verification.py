from __future__ import annotations

import csv
from pathlib import Path
from typing import Iterable, Sequence


# The first five columns are the handoff's required schema; the rest document each checkpoint.
METRIC_FIELDS = [
    "model",
    "step",
    "congressional_loss",
    "general_loss",
    "mean_seed_drift",
    "max_seed_drift",
    "loss_metric",
    "epoch",
    "train_loss",
    "tokens_seen",
    "elapsed_seconds",
    "flags",
    "checkpoint",
    "event",
]


def append_metric_row(path: str | Path, row: dict[str, float | int | str]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists() or path.stat().st_size == 0
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=METRIC_FIELDS, extrasaction="ignore")
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


def _number(value: object) -> float | None:
    return None if value is None or value == "" else float(value)


def check_stop_conditions(
    baseline: dict,
    previous: dict,
    current: dict,
    general_rel_increase: float,
    drift_threshold: float,
) -> list[str]:
    """Return the handoff's stop flags for a checkpoint row (empty list = keep going).

    General loss is compared against the step-0 value of the same model; seed drift flags only when
    it is above the threshold and still higher than at the previous checkpoint.
    """
    flags: list[str] = []
    base, now = _number(baseline.get("general_loss")), _number(current.get("general_loss"))
    if base and now is not None and (now - base) / base > general_rel_increase:
        flags.append(
            f"general-domain loss rose {(now - base) / base:.1%} over step 0 ({base:.4f} -> {now:.4f}); "
            f"limit is {general_rel_increase:.0%}"
        )
    drift, prev_drift = _number(current.get("mean_seed_drift")), _number(previous.get("mean_seed_drift"))
    if drift is not None and prev_drift is not None and drift > drift_threshold and drift > prev_drift:
        flags.append(
            f"mean seed drift {drift:.4f} is above {drift_threshold:.4f} and still climbing (previous {prev_drift:.4f})"
        )
    return flags


def assert_checkpoint_conditions(model: str, row: dict[str, float | int | str]) -> None:
    """Fail fast when the verification signal indicates a problematic training run."""
    general_loss = float(row["general_loss"])
    drift = float(row["mean_seed_drift"])
    if general_loss > 2.0:
        raise RuntimeError(f"{model}: general-domain loss increased sharply; stop and inspect training.")
    if drift > 0.5:
        raise RuntimeError(f"{model}: mean seed drift is too large and still climbing; stop and report.")
