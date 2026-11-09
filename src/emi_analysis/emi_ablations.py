"""Ablations of contextual EMI on the held-out speeches, for one emi_score output (one model and representation).

    python -m emi_analysis.emi_ablations --scores results/emi/raw/bert/run-job4757959/step350847 \\
        [--compare <other emi_score outputs> ...]

Every variant re-scores the held-out speeches (never a source of keyword occurrences) and is compared with the main
held-out EMI of the same scores folder (its primary layer, corpus mean, main poles):
    context/alone           poles from each keyword run on its own (the paper's way) instead of in context
    era_poles/<e>           poles from one era's occurrences only (drift; their cosines in era_pole_cosines.csv)
    sampling/unbalanced     the same number of occurrences per keyword drawn from all eras at once
    n_per_era/<n>           25, 50, 200 occurrences per keyword and era instead of the main number
    min_per_era/1           every era with any occurrence counts toward a keyword (no minimum)
    keywords/all_eras       only keywords with at least 100 occurrences in every era (no modern-vocabulary tilt)
    loo/<word>              the poles without one keyword (loo_keywords.csv)
    scoring/projection      one cosine with P_ev - P_int instead of two cosines
    scoring/no_length_adjust, scoring/no_centering
    series/chunk_weighted   Congress means weighted by the paper's 150-word chunk count
    layers/<L>              every other layer of the same scores folder
    compare/<label>         other scores folders (other representations, the base model, other SAE seeds or configs)
Each row: pearson_speech, spearman_speech (held-out speech EMI) and pearson_series (per-Congress means), n_scored.

Output in results/emi/ablations/<representation>/<model>/<run>/step<N>/layer<L>/:
    ablations.csv, loo_keywords.csv, era_pole_cosines.csv, ablation_series.csv, meta.json
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import time
import warnings
from argparse import Namespace
from datetime import datetime
from pathlib import Path

import numpy as np

from .emi_common import N_ERAS, n_chunks_150, scorable
from .emi_score import (aligned_stats, congress_series, cosines, emi_from_cosines, keyword_vectors, load_representation,
                        poles, projection, sae_name, standardize)
from .runinfo import banner, format_duration, setup_logging

logger = logging.getLogger("emi_analysis")

SAMPLE_SIZES = (25, 50, 200)
ALL_ERAS_MIN = 100


def _corr(a: np.ndarray, b: np.ndarray) -> tuple[float, float, int]:
    from scipy.stats import pearsonr, spearmanr

    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 3 or np.std(a[ok]) == 0 or np.std(b[ok]) == 0:
        return float("nan"), float("nan"), int(ok.sum())
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return float(pearsonr(a[ok], b[ok])[0]), float(spearmanr(a[ok], b[ok])[0]), int(ok.sum())


def _series_means(rows: list[dict]) -> dict[int, float]:
    return {r["congress"]: r["mean"] for r in rows}


def _cos(a: np.ndarray, b: np.ndarray) -> float:
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))


def rep_tag(meta: dict) -> str:
    return meta["representation"] if meta["representation"] == "raw" else f"{meta['representation']}/{sae_name(meta['sae'])}"


def heldout_emi_of(scores: Path, speech_ids: list[str], layer: int | None = None) -> np.ndarray:
    """Held-out EMI from another scores folder's stored cosines, aligned to the given speech IDs and z-scored on them."""
    meta = json.loads((scores / "meta.json").read_text(encoding="utf-8"))
    layer = meta["primary_layer"] if layer is None else layer
    act = Path(meta["activations"])
    ids = (act / "speech_ids.txt").read_text(encoding="utf-8").split()
    position = {s: i for i, s in enumerate(ids)}
    rows = np.asarray([position.get(s, -1) for s in speech_ids])
    stats = aligned_stats(act, Path(meta["occurrences"]))
    ev = np.load(scores / f"layer_{layer:02d}_ev.npy", mmap_mode="r")
    intu = np.load(scores / f"layer_{layer:02d}_int.npy", mmap_mode="r")
    found = rows >= 0
    ev_h, int_h = np.full(len(rows), np.nan), np.full(len(rows), np.nan)
    ev_h[found], int_h[found] = ev[rows[found]], intu[rows[found]]
    wc = np.zeros(len(rows), dtype=np.int64)
    top = np.zeros(len(rows))
    wc[found], top[found] = stats["word_count"][rows[found]], stats["top100_frac"][rows[found]]
    emi, _ = emi_from_cosines(ev_h, int_h, wc, found & scorable(wc, top))
    return emi


def build_cli() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Held-out ablations of contextual EMI for one scores folder.")
    parser.add_argument("--scores", required=True, help="emi_score output folder.")
    parser.add_argument("--layer", type=int, default=None, help="Layer to ablate (default: the primary layer).")
    parser.add_argument("--compare", action="append", default=[], help="Other emi_score folders to compare with.")
    parser.add_argument("--n-boot", type=int, default=200)
    parser.add_argument("--output-root", default="results/emi/ablations")
    return parser


def run(args: argparse.Namespace) -> dict:
    started = time.time()
    scores = Path(args.scores)
    meta = json.loads((scores / "meta.json").read_text(encoding="utf-8"))
    layer = meta["primary_layer"] if args.layer is None else args.layer
    checkpoint = meta["checkpoint"]
    out = Path(args.output_root) / rep_tag(meta) / meta["model"] / checkpoint["run"] / f"step{checkpoint['step']}" / f"layer{layer:02d}"
    out.mkdir(parents=True, exist_ok=True)
    setup_logging(out / "ablations.log", mode="a")

    act, occ_dir, kw_dir = Path(meta["activations"]), Path(meta["occurrences"]), Path(meta["keywords"])
    rep = load_representation(Namespace(activations=str(act), keywords=str(kw_dir), representation=meta["representation"],
                                        sae=meta["sae"], sae_features=meta["sae_features"]), layer)
    stats = aligned_stats(act, occ_dir)
    held = np.flatnonzero(stats["heldout"])
    if len(held) == 0:
        raise RuntimeError(f"{act} holds no held-out speeches to ablate on.")
    speech_ids = (act / "speech_ids.txt").read_text(encoding="utf-8").split()
    held_ids = [speech_ids[i] for i in held]
    wc, congress = stats["word_count"][held], stats["congress"][held]
    ok = scorable(wc, stats["top100_frac"][held])
    occ = np.load(occ_dir / "occurrences.npz")
    keywords = json.loads((occ_dir / "keywords.json").read_text(encoding="utf-8"))
    words, kw_pole = keywords["words"], np.asarray(keywords["poles"], dtype=np.int8)
    mu = np.load(scores / f"mu_layer_{layer:02d}.npy")
    main_poles = np.load(scores / f"poles_layer_{layer:02d}.npz")
    n_main = int(meta["n_per_era"])
    min_main = int(meta.get("min_per_era", 1))
    banner(logger, f"EMI ABLATIONS: {meta['model']} ({rep_tag(meta)}), layer {layer}", {
        "scores": scores, "held-out speeches": f"{len(held):,} ({int(ok.sum()):,} scorable)", "output": out})

    def emi_for(p_ev, p_int, center=True, adjust=True):
        ev, intu = cosines(rep.source, held, mu, p_ev, p_int, encode=rep.encode, center=center)
        return emi_from_cosines(ev, intu, wc, ok, adjust)[0]

    def poles_from(use, balanced=True, keep=None, min_per_era=None):
        vectors, _ = keyword_vectors(rep.occ, occ["keyword"], occ["era"], len(words), use, balanced,
                                     min_per_era=min_main if min_per_era is None else min_per_era)
        if keep is not None:
            vectors = np.where(keep[:, None], vectors, np.nan)
        if not (np.isfinite(vectors).all(axis=1) & (kw_pole == 0)).any() or not (np.isfinite(vectors).all(axis=1) & (kw_pole == 1)).any():
            return None, vectors
        return poles(vectors, kw_pole, mu), vectors

    primary = emi_for(main_poles["p_ev"], main_poles["p_int"])
    primary_series = congress_series(primary, congress, n_boot=args.n_boot)
    primary_means = _series_means(primary_series)
    rows, loo_rows, series_rows = [], [], [{"variant": "primary", **r} for r in primary_series]

    def record(variant: str, emi: np.ndarray | None, weights=None, table=rows):
        if emi is None:
            table.append({"variant": variant, "pearson_speech": float("nan"), "spearman_speech": float("nan"),
                          "pearson_series": float("nan"), "n_scored": 0, "note": "a pole has no keywords"})
            return
        series = congress_series(emi, congress, weights=weights, n_boot=args.n_boot)
        means = _series_means(series)
        common = sorted(set(means) & set(primary_means))
        pr, sp, n = _corr(primary, emi)
        ps, _, _ = _corr(np.asarray([primary_means[c] for c in common]), np.asarray([means[c] for c in common]))
        table.append({"variant": variant, "pearson_speech": pr, "spearman_speech": sp, "pearson_series": ps, "n_scored": n, "note": ""})
        series_rows.extend({"variant": variant, **r} for r in series)

    use_main = (occ["rank_balanced"] >= 0) & (occ["rank_balanced"] < n_main)

    # context: the keyword on its own
    if rep.name == "raw":
        alone = np.load(kw_dir / f"alone_layer_{layer:02d}.npy")
    elif rep.name == "speech_sae":
        alone = rep.encode(np.load(kw_dir / f"alone_layer_{layer:02d}.npy"))
    else:
        alone = np.load(kw_dir / f"alone_sae_{rep.label}.npy")
    record("context/alone", emi_for(*poles(alone.astype(np.float64), kw_pole, mu)))

    # drift: one era's poles
    era_poles = {"pooled": (main_poles["p_ev"], main_poles["p_int"])}
    for e in range(N_ERAS):
        p, _ = poles_from(use_main & (occ["era"] == e))
        record(f"era_poles/{e}", emi_for(*p) if p else None)
        if p:
            era_poles[str(e)] = p
    cosine_rows = []
    names = list(era_poles)
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            (ea, ia), (eb, ib) = era_poles[a], era_poles[b]
            cosine_rows += [{"era_a": a, "era_b": b, "pole": "evidence", "cosine": _cos(ea, eb)},
                            {"era_a": a, "era_b": b, "pole": "intuition", "cosine": _cos(ia, ib)},
                            {"era_a": a, "era_b": b, "pole": "difference", "cosine": _cos(ea - ia, eb - ib)}]

    # sampling
    p, _ = poles_from(occ["rank_unbalanced"] >= 0, balanced=False)
    record("sampling/unbalanced", emi_for(*p) if p else None)
    for n in SAMPLE_SIZES:
        p, _ = poles_from((occ["rank_balanced"] >= 0) & (occ["rank_balanced"] < n))
        record(f"n_per_era/{n}", emi_for(*p) if p else None)

    p, _ = poles_from(use_main, min_per_era=1)
    record("min_per_era/1", emi_for(*p) if p else None)
    sampled = np.zeros((len(words), N_ERAS), dtype=np.int64)
    in_sample = occ["rank_balanced"] >= 0
    np.add.at(sampled, (occ["keyword"][in_sample], occ["era"][in_sample]), 1)
    all_eras = (sampled >= ALL_ERAS_MIN).all(axis=1)
    logger.info(f"keywords with >= {ALL_ERAS_MIN} occurrences in every era: {int((all_eras & (kw_pole == 0)).sum())} evidence, "
                f"{int((all_eras & (kw_pole == 1)).sum())} intuition")
    p, _ = poles_from(use_main, keep=all_eras)
    record("keywords/all_eras", emi_for(*p) if p else None)

    # leave one keyword out
    main_vectors = main_poles["keyword_vectors"]
    for k in np.flatnonzero(np.isfinite(main_vectors).all(axis=1)):
        keep = np.ones(len(words), dtype=bool)
        keep[k] = False
        others = np.isfinite(main_vectors).all(axis=1) & keep
        if not (others & (kw_pole == 0)).any() or not (others & (kw_pole == 1)).any():
            record(f"loo/{words[k]}", None, table=loo_rows)
            continue
        record(f"loo/{words[k]}", emi_for(*poles(main_vectors, kw_pole, mu, keep)), table=loo_rows)

    # scoring details
    proj = projection(rep.source, held, mu, main_poles["p_ev"], main_poles["p_int"], encode=rep.encode)
    record("scoring/projection", standardize(proj, wc, ok)[0])
    record("scoring/no_length_adjust", emi_for(main_poles["p_ev"], main_poles["p_int"], adjust=False))
    record("scoring/no_centering", emi_for(main_poles["p_ev"] + mu, main_poles["p_int"] + mu, center=False))
    record("series/chunk_weighted", primary, weights=n_chunks_150(wc))

    # other layers of the same folder, and other folders
    for other in meta["layers"]:
        record(f"layers/{other}", heldout_emi_of(scores, held_ids, other))
    for path in args.compare:
        other_meta = json.loads((Path(path) / "meta.json").read_text(encoding="utf-8"))
        label = f"{rep_tag(other_meta)}/{other_meta['model']}/{other_meta['checkpoint']['run']}/layer{other_meta['primary_layer']}"
        record(f"compare/{label}", heldout_emi_of(Path(path), held_ids))

    fields = ["variant", "pearson_speech", "spearman_speech", "pearson_series", "n_scored", "note"]
    for name, table in (("ablations.csv", rows), ("loo_keywords.csv", loo_rows)):
        with (out / name).open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(table)
    with (out / "era_pole_cosines.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["era_a", "era_b", "pole", "cosine"])
        writer.writeheader()
        writer.writerows(cosine_rows)
    with (out / "ablation_series.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["variant", "congress", "n", "mean", "lower", "upper", "ci_method"])
        writer.writeheader()
        writer.writerows(series_rows)
    (out / "meta.json").write_text(json.dumps({
        "scores": str(scores), "layer": layer, "representation": meta["representation"], "model": meta["model"],
        "n_heldout": int(len(held)), "n_scored": int(ok.sum()), "compare": args.compare,
        "finished_at": datetime.now().isoformat(timespec="seconds"), "elapsed_seconds": round(time.time() - started, 1),
    }, indent=2) + "\n", encoding="utf-8")
    for row in rows:
        logger.info(f"{row['variant']:<40} speech r {row['pearson_speech']:+.3f} | rho {row['spearman_speech']:+.3f} | "
                    f"series r {row['pearson_series']:+.3f} | n {row['n_scored']:,}")
    loo = [r["pearson_speech"] for r in loo_rows if np.isfinite(r["pearson_speech"])]
    banner(logger, f"EMI ABLATIONS COMPLETE: {meta['model']} ({rep_tag(meta)}), layer {layer}", {
        "variants": len(rows), "leave-one-keyword-out": f"{len(loo_rows)} keywords, lowest speech r {min(loo):.3f}" if loo else "none",
        "output": out, "elapsed": format_duration(time.time() - started)})
    return {"output_dir": str(out)}


def main(argv: list[str] | None = None) -> dict:
    args = build_cli().parse_args(argv)
    try:
        return run(args)
    except Exception:
        logger.exception("EMI ablations failed")
        raise


if __name__ == "__main__":
    main()
