"""Re-choose the primary layer of finished emi_score results with the speech-level rule, without re-scoring.

    python -m emi_analysis.emi_select_layer results/emi/raw/bert/run-job4757959/step350847 [more result folders]

emi_score now picks the primary layer by leave-one-keyword-out speech AUC (emi_score.speech_layer_auc): poles from
the other keywords score the whole speeches a held-out keyword was sampled from, as length-adjusted, z-scored EMI.
Results scored before that rule picked the layer from the keyword-token AUC instead. This command recomputes only
the speech AUC per layer from the saved mu_layer_XX.npy / params_layer_XX.json and the speech activations, then:

    layer_selection.csv   gains loo_speech_auc, n_auc_speeches, n_auc_keywords (old columns kept as they were)
    meta.json             primary_layer / primary_rule updated; the replaced choice kept under previous_primary
    select_layer.log      the per-layer AUCs and the decision

Scores (layer_XX_*.npy, series.csv) are untouched: every layer is already scored.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import time
from datetime import datetime
from pathlib import Path

import numpy as np

from .emi_common import scorable
from .emi_score import (PRIMARY_RULE, _collection_index, _select_stats, choose_primary, load_representation,
                        occurrence_rows, speech_layer_auc, write_selection)
from .runinfo import banner, environment_info, format_duration, setup_logging

logger = logging.getLogger("emi_analysis")


def select(out: Path) -> dict:
    started = time.time()
    meta = json.loads((out / "meta.json").read_text(encoding="utf-8"))
    setup_logging(out / "select_layer.log", mode="a")
    act, occ_dir = Path(meta["activations"]), Path(meta["occurrences"])
    occ = np.load(occ_dir / "occurrences.npz")
    corpus_stats = dict(np.load(occ_dir / "speech_stats.npz"))
    index = _collection_index(act, occ_dir)
    stats = _select_stats(corpus_stats, index)
    occ_rows = occurrence_rows(occ["row"], index, len(corpus_stats["congress"]))
    n_keywords = len(json.loads((occ_dir / "keywords.json").read_text(encoding="utf-8"))["words"])
    ok = scorable(stats["word_count"], stats["top100_frac"])
    use = (occ["rank_balanced"] >= 0) & (occ["rank_balanced"] < meta["n_per_era"])
    with (out / "layer_selection.csv").open(newline="", encoding="utf-8") as handle:
        old = {int(r["layer"]): r for r in csv.DictReader(handle)}
    rep_args = argparse.Namespace(representation=meta["representation"], activations=meta["activations"],
                                  keywords=meta["keywords"], sae=meta.get("sae"), sae_features=meta.get("sae_features"))
    banner(logger, f"LAYER SELECTION (speech AUC): {meta['model']} ({meta['representation']})", {
        "results": out, "activations": act, "occurrences": f"{occ_dir} ({int(use.sum()):,} sampled occurrences)",
        "current primary": f"{meta['primary_layer']} ({meta['primary_rule']})", "layers": ", ".join(map(str, meta["layers"])),
        **environment_info(),
    })

    selection = []
    for layer in meta["layers"]:
        t0 = time.time()
        rep = load_representation(rep_args, layer)
        mu = np.load(out / f"mu_layer_{layer:02d}.npy")
        params = json.loads((out / f"params_layer_{layer:02d}.json").read_text(encoding="utf-8"))
        speech = speech_layer_auc(rep.source, rep.occ, occ_rows, occ["keyword"], occ["pole"], occ["era"], mu, use,
                                  n_keywords, ok, stats["word_count"], params, meta["min_per_era"], rep.encode)
        prev = old.get(layer, {})
        selection.append({"layer": layer, "loo_speech_auc": speech["auc"], "n_auc_speeches": speech["n_speeches"],
                          "n_auc_keywords": speech["n_keywords"], "loo_keyword_auc": prev.get("loo_keyword_auc", ""),
                          "pole_cosine": prev.get("pole_cosine", ""), "missing_keywords": prev.get("missing_keywords", "")})
        logger.info(f"layer {layer:2d}: speech AUC {speech['auc']:.3f} ({speech['n_speeches']:,} speeches, "
                    f"{speech['n_keywords']} keywords, {speech['n_excluded_shared']:,} shared skipped) | "
                    f"keyword-token AUC {float(prev.get('loo_keyword_auc', 'nan')):.3f} | {format_duration(time.time() - t0)}")

    write_selection(out / "layer_selection.csv", selection)
    primary = choose_primary(selection, meta["layers"])
    if meta.get("primary_rule") != PRIMARY_RULE and "previous_primary" not in meta:  # keep the original choice
        meta["previous_primary"] = {"primary_layer": meta["primary_layer"], "primary_rule": meta["primary_rule"]}
    previous = meta["primary_layer"]
    meta.update(primary_layer=int(primary), primary_rule=PRIMARY_RULE,
                layer_selection_updated_at=datetime.now().isoformat(timespec="seconds"))
    (out / "meta.json").write_text(json.dumps(meta, indent=2, default=str) + "\n", encoding="utf-8")
    best = next(s for s in selection if s["layer"] == primary)
    banner(logger, f"LAYER SELECTION COMPLETE: {meta['model']} ({meta['representation']})", {
        "primary layer": f"{primary} (speech AUC {best['loo_speech_auc']:.3f}, {best['n_auc_speeches']:,} speeches)",
        "was": previous, "meta.json": out / "meta.json", "elapsed": format_duration(time.time() - started),
    })
    return {"output_dir": str(out), "primary_layer": int(primary), "previous_primary_layer": int(previous)}


def build_cli() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Re-choose the primary layer of emi_score results by speech AUC.")
    parser.add_argument("results", nargs="+", help="emi_score output folders (each with meta.json).")
    return parser


def main(argv: list[str] | None = None) -> list[dict]:
    args = build_cli().parse_args(argv)
    try:
        return [select(Path(path)) for path in args.results]
    except Exception:
        logger.exception("Layer selection failed")
        raise


if __name__ == "__main__":
    main()
