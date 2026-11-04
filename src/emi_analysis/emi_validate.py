"""Validation of contextual EMI against the paper's Word2Vec series and the Prolific human ratings (run last).

    # 1. the rated excerpts as a speech file (then collect their activations like any speeches)
    python -m emi_analysis.emi_validate export --prolific <EMI repo>/data/prolific_ratings_US_congress_congressw2v.csv \\
        --output data/validation/prolific_excerpts.jsonl
    # 2. per scores folder
    python -m emi_analysis.emi_validate score --scores results/emi/raw/bert/run-job4757959/step350847 \\
        --excerpt-activations activations/bert/run-job4757959/step350847/prolific \\
        --prolific <...>.csv --w2v <EMI repo>/data/congress_EMI_w2v_bootstrap_CIs.csv

Trend: Pearson/Spearman between the per-Congress mean EMI (primary layer) and the W2V series (years mapped to
Congresses with the paper's formula). Human: the rated items are ~150-word excerpts, so the excerpts themselves are
scored with the corpus poles, mean and length/z parameters, and correlated with the mean over raters of
evidence_based - evidence_free; the file's own W2V score is correlated with the same target as a reference.
Nothing upstream reads these results.

Output in results/emi/validation/<representation>/<model>/<run>/step<N>/: validation.json, excerpt_scores.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
from argparse import Namespace
from datetime import datetime
from pathlib import Path

import numpy as np

from .emi_ablations import _corr, rep_tag
from .emi_common import top100_words, word_stats
from .emi_score import apply_standardize, cosines, load_representation
from .runinfo import banner, setup_logging

logger = logging.getLogger("emi_analysis")


def congress_of_year(year) -> np.ndarray:
    """The paper's mapping: congress = int((year - 1789) / 2 + 1)."""
    return (((np.asarray(year, dtype=np.float64) - 1789) / 2) + 1).astype(np.int64)


def excerpt_targets(prolific: str | Path) -> list[dict]:
    """One row per rated excerpt: mean over raters of evidence_based - evidence_free."""
    import pandas as pd

    df = pd.read_csv(prolific)
    df["target"] = df["evidence_based"].astype(float) - df["evidence_free"].astype(float)
    rows = []
    for item, group in df.groupby("item_id", sort=True):
        first = group.iloc[0]
        rows.append({"speech_id": f"item{item}", "target": float(group["target"].mean()), "n_raters": int(len(group)),
                     "congress": int(congress_of_year(first["starting_year"])), "text": str(first["text"]),
                     "w2v_emi": float(first["evidence_minus_intuition_score"]), "source_speech_id": str(first["speech_id"])})
    return rows


def export(args) -> dict:
    rows = excerpt_targets(args.prolific)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("".join(json.dumps({"speech_id": r["speech_id"], "congress": r["congress"], "text": r["text"]}) + "\n"
                           for r in rows), encoding="utf-8")
    return {"output": str(out), "n_items": len(rows)}


def score(args) -> dict:
    scores = Path(args.scores)
    meta = json.loads((scores / "meta.json").read_text(encoding="utf-8"))
    layer = meta["primary_layer"]
    checkpoint = meta["checkpoint"]
    out = Path(args.output_root) / rep_tag(meta) / meta["model"] / checkpoint["run"] / f"step{checkpoint['step']}"
    out.mkdir(parents=True, exist_ok=True)
    setup_logging(out / "validate.log", mode="a")
    report: dict = {"scores": str(scores), "layer": layer, "representation": meta["representation"], "model": meta["model"]}

    if args.w2v:
        import pandas as pd

        w2v = pd.read_csv(args.w2v)
        w2v_by_congress = dict(zip(congress_of_year(w2v["year"]).tolist(), w2v["evidence_minus_intuition_score"].astype(float)))
        with (scores / "series.csv").open(newline="", encoding="utf-8") as handle:
            ours = {int(r["congress"]): float(r["mean"]) for r in csv.DictReader(handle) if int(r["layer"]) == layer}
        common = sorted(c for c in set(ours) & set(w2v_by_congress) if np.isfinite(ours[c]))
        pr, sp, n = _corr(np.asarray([ours[c] for c in common]), np.asarray([w2v_by_congress[c] for c in common]))
        report["trend"] = {"pearson": pr, "spearman": sp, "n_congresses": n}

    if args.excerpt_activations:
        exc = Path(args.excerpt_activations)
        exc_meta = json.loads((exc / "meta.json").read_text(encoding="utf-8"))
        rep = load_representation(Namespace(activations=str(exc), keywords=meta["keywords"], representation=meta["representation"],
                                            sae=meta["sae"], sae_features=str(exc)), layer)
        mu = np.load(scores / f"mu_layer_{layer:02d}.npy")
        p = np.load(scores / f"poles_layer_{layer:02d}.npz")
        params = json.loads((scores / f"params_layer_{layer:02d}.json").read_text(encoding="utf-8"))
        ev, intu = cosines(rep.source, None, mu, p["p_ev"], p["p_int"], encode=rep.encode)
        ids = (exc / "speech_ids.txt").read_text(encoding="utf-8").split()
        texts = {}
        for path in exc_meta["inputs"]:
            with open(path, encoding="utf-8") as handle:
                for line in handle:
                    if line.strip():
                        row = json.loads(line)
                        texts[str(row["speech_id"])] = row["text"]
        top100 = top100_words()
        wc = np.asarray([word_stats(texts[i], top100)[0] for i in ids])
        emi = apply_standardize(ev, wc, params["ev"]) - apply_standardize(intu, wc, params["int"])
        targets = {t["speech_id"]: t for t in excerpt_targets(args.prolific)}
        keep = [i for i, s in enumerate(ids) if s in targets]
        target = np.asarray([targets[ids[i]]["target"] for i in keep])
        ours = emi[keep]
        w2v_scores = np.asarray([targets[ids[i]]["w2v_emi"] for i in keep])
        pr, sp, n = _corr(ours, target)
        wpr, wsp, _ = _corr(w2v_scores, target)
        report["human"] = {"pearson": pr, "spearman": sp, "n_items": n, "w2v_pearson": wpr, "w2v_spearman": wsp,
                           "ours_vs_w2v_pearson": _corr(ours, w2v_scores)[0]}
        with (out / "excerpt_scores.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=["speech_id", "emi", "target", "w2v_emi", "word_count"])
            writer.writeheader()
            for j, i in enumerate(keep):
                writer.writerow({"speech_id": ids[i], "emi": float(ours[j]), "target": float(target[j]),
                                 "w2v_emi": float(w2v_scores[j]), "word_count": int(wc[i])})

    report["finished_at"] = datetime.now().isoformat(timespec="seconds")
    (out / "validation.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    banner(logger, f"EMI VALIDATION: {meta['model']} ({rep_tag(meta)}), layer {layer}", {
        **({"trend vs W2V (Pearson / Spearman)": f"{report['trend']['pearson']:+.3f} / {report['trend']['spearman']:+.3f} "
                                                  f"over {report['trend']['n_congresses']} Congresses"} if "trend" in report else {}),
        **({"human ratings (Pearson / Spearman)": f"{report['human']['pearson']:+.3f} / {report['human']['spearman']:+.3f} "
                                                  f"over {report['human']['n_items']} excerpts",
            "W2V reference vs human": f"{report['human']['w2v_pearson']:+.3f} / {report['human']['w2v_spearman']:+.3f}"}
           if "human" in report else {}),
        "output": out})
    return report


def build_cli() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Validate contextual EMI against the W2V series and human ratings.")
    sub = parser.add_subparsers(dest="command", required=True)
    exp = sub.add_parser("export", help="Write the rated excerpts as a speech JSONL.")
    exp.add_argument("--prolific", required=True)
    exp.add_argument("--output", default="data/validation/prolific_excerpts.jsonl")
    sc = sub.add_parser("score", help="Correlate one scores folder with the validation targets.")
    sc.add_argument("--scores", required=True)
    sc.add_argument("--excerpt-activations", default=None, help="Collection of the exported excerpts (raw, or token-SAE).")
    sc.add_argument("--prolific", default=None)
    sc.add_argument("--w2v", default=None)
    sc.add_argument("--output-root", default="results/emi/validation")
    return parser


def main(argv: list[str] | None = None) -> dict:
    args = build_cli().parse_args(argv)
    if args.command == "export":
        return export(args)
    try:
        return score(args)
    except Exception:
        logger.exception("EMI validation failed")
        raise


if __name__ == "__main__":
    main()
