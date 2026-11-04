# Contextual EMI Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Score EMI for every speech from raw activations, a token-level SAE and a speech-level SAE, with
era-balanced keyword-occurrence poles, plus held-out ablations and validation.

**Architecture:** One shared occurrence sample (CPU) → per-model keyword activations (GPU) → a scoring library
used by every representation (`emi_score`) → SAE training (`sae`, `train_sae`) → token-SAE corpus pass
(an `--sae` mode of the existing collector) → ablations and validation drivers. Each stage has a CLI, a
SLURM script, `meta.json` + banner logs, and tests on tiny models.

**Tech Stack:** Python 3.10-compatible, torch, transformers, datasets, numpy, scipy, wordfreq.

**Spec:** `docs/superpowers/specs/2026-11-04-contextual-emi-design.md`

## Global Constraints

- Code must run on Python 3.10 (Quest) and the local 3.14 venv.
- SLURM: account `p53299`, partition `gengpu` with `--gres=gpu:h100:1` for GPU stages, mail to
  `bettencourt@u.northwestern.edu` on BEGIN/END/FAIL, logs in `logs/<name>_%j.out/.err`.
- Eras: 46–54, 55–63, 64–72, 73–81, 82–90, 91–99, 100–108, 109–117.
- Occurrence sample sizes: 200 ranked per keyword-era (main uses the first 100; ablation 9 uses 25/50/100/200);
  unbalanced ablation: 800 per keyword. At most one occurrence per speech per keyword. Training speeches only.
- Scoring filters: ≤ 10 words or < 5% top-100 words → unscored; 10-word length bins; z-scores; EMI = z_ev − z_int.
- SAE primary config: TopK, k = 64, width 8×, seeds 0 and 1; token-SAE ≈ 200M era-balanced training tokens.
- ModernBERT forward passes in strict fp32 everywhere; others bf16.
- Nothing from the old `text-generation-project` outputs is read.

## Review Focus

1. A keyword at the very start/end of a chunk, or a span that straddles a chunk boundary → mapped to the chunk
   of its first token, counted in the log (test in Task 2).
2. Byte-level BPE leading spaces (GPT-2/Qwen3 `Ġfact`) → the token overlapping the word's characters is used
   (test in Task 2, gpt2 + qwen3).
3. A keyword with zero occurrences in some era → era skipped in its average, logged; a keyword with none at all
   → excluded from the pole and reported (test in Task 3).
4. All-unscorable Congress or a length bin with one speech → series row with n = 0 / bin mean equals the score,
   no NaN crash (test in Task 3).
5. Resuming an interrupted token-SAE corpus pass, or mixing SAEs on resume → resumes / refuses without
   `--overwrite` (test in Task 6).

---

### Task 1: Shared occurrence sample (`keyword_occurrences.py`)

**Files:** Create `src/emi_analysis/emi_common.py`, `src/emi_analysis/keyword_occurrences.py`,
`tests/test_keyword_occurrences.py`; add `wordfreq>=3.0` and `scipy>=1.10` to `requirements.txt`.

**Interfaces — produces:**
- `emi_common.ERAS: list[tuple[int, int]]`, `era_of(congress: np.ndarray) -> np.ndarray` (−1 outside).
- `emi_common.load_poles_lexicons(config) -> tuple[list[str], list[str]]` (wraps `train.load_lexicon`).
- `emi_common.paper_preprocess(text) -> str`, `word_stats(text, top100) -> tuple[int, float]`
  (word count and top-100 fraction exactly as `compute_sbert_avg_lexicon.py`).
- `emi_common.n_chunks_150(word_count: np.ndarray) -> np.ndarray` (paper chunking: 150-word chunks, a last
  chunk < 50 words merged).
- Output dir `emi/occurrences/<tag>/`: `occurrences.npz` with arrays `row, keyword (index), pole (0 ev / 1 int),
  era, start, end, rank_balanced (−1 if not in), rank_unbalanced (−1 if not in)`; `speech_stats.npz` with
  `word_count, top100_frac, heldout (bool), congress`; `keywords.json` (ordered words + poles); `meta.json`
  (counts per keyword-era, shortfalls, flagged < 20 words).
- CLI: `--config`, `--input` (repeat, default train + held-out in that order), `--heldout` path,
  `--per-era 200`, `--unbalanced 800`, `--seed 0`, `--output-root emi/occurrences`, `--tag corpus`, `--num-proc`.

- [ ] Tests: `test_sample_is_era_balanced_one_per_speech_and_excludes_heldout`,
  `test_sample_is_deterministic_and_prefix_ranked`, `test_word_stats_match_paper_rules`,
  `test_chunk_count_merges_short_last_chunk` (150→1, 199→1, 200→2, 301→2, 350→3).
- [ ] Implement: case-insensitive whole-word regex `(?<![\w'])(w1|w2|…)(?![\w'])`; per (speech, keyword) pick one
  match by a splitmix64 hash of (seed, row, keyword); bottom-k selection by a second hash per keyword-era
  (balanced) and per keyword (unbalanced) so ranks are deterministic and prefixes are valid smaller samples.
- [ ] Run tests, commit.

### Task 2: Keyword activations (`keyword_activations.py`)

**Files:** Create `src/emi_analysis/keyword_activations.py`, `tests/test_keyword_activations.py`.

**Interfaces — consumes:** Task 1 outputs; `collect_activations.choose_checkpoint`, `train.load_model_and_tokenizer`,
`train.collate`, `train.autocast_for`. **Produces** `emi/keywords/<model>/<run>/step<N>/<occurrence tag>/`:
`occ_layer_XX.npy` (n_occ × hidden, float32, row order = `occurrences.npz`), `alone_layer_XX.npy`
(n_keywords × hidden), optional `occ_sae_<name>.npy` (mean over the keyword's pieces of SAE(piece), with
`--sae`), `meta.json` (checkpoint, precision, straddling spans, missing spans).
- CLI: `--model`, `--config`, `--run-dir | --checkpoint | --base`, `--occurrences DIR`, `--precision`,
  `--sae PATH` (repeatable), `--output-root emi/keywords`, `--batch-chunks 32`.

- [ ] Tests: `test_occurrence_state_equals_token_state_in_its_chunk` (bert, gpt2, qwen3 tiny; compare with a manual
  forward of the chunk and the overlapping token positions), `test_alone_vectors_exclude_special_tokens`,
  `test_straddling_span_uses_first_tokens_chunk`.
- [ ] Implement: tokenize each needed speech with `return_offsets_mapping=True`, window = max_length − wrappers,
  chunk index = token // window; run unique chunks once, batched by length; mean the overlapping positions.
- [ ] Run tests, commit.

### Task 3: Scoring library and raw scoring (`emi_score.py`)

**Files:** Create `src/emi_analysis/emi_score.py`, `tests/test_emi_score.py`.

**Interfaces — produces:**
- `token_weighted_mean(source, counts, rows=None) -> np.ndarray` (float64).
- `keyword_vectors(occ: np.ndarray, keyword, era, n_keywords, use: np.ndarray, balanced=True) -> (np.ndarray, list[int])`
  (per-keyword mean of per-era means; returns missing keywords).
- `poles(kw_vectors, kw_pole, mu, keep: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]` (centered).
- `cosines(source, rows, mu, p_ev, p_int, encode=None, center=True) -> tuple[np.ndarray, np.ndarray]`, block-streamed.
- `projection(...)` same signature, returns one score.
- `emi_from_cosines(ev, intu, word_count, scorable, adjust=True) -> (emi, params)` where params hold bin means
  and z mean/sd for reuse on new texts; `apply_params(ev, intu, word_count, params)`.
- `congress_series(emi, congress, weights=None, n_boot=1000, seed=0) -> list[dict]`.
- `layer_auc(occ, keyword, pole, era, mu, use) -> float` (leave one keyword out).
- Representations: `Representation` = `raw` | `speech_sae` (encode stored vectors with a speech SAE) | `token_sae`
  (features from a token-SAE collection; occurrence vectors from `occ_sae_<name>.npy`).
- CLI: `--model`, `--activations DIR`, `--keywords DIR`, `--occurrences DIR`, `--representation`,
  `--sae PATH`, `--sae-features DIR`, `--layers all|<int,…>`, `--n-per-era 100`, `--output-root results/emi`.
  Output `results/emi/<representation>/<model>/<run>/`: `mu_layer_XX.npy`, `poles_layer_XX.npz`,
  `layer_XX_{ev,int,emi}.npy` (float32, NaN unscored), `params_layer_XX.json`, `series.csv`,
  `layer_selection.csv`, `meta.json` with `primary_layer`.

- [ ] Tests: `test_token_weighted_mean_equals_mean_over_tokens`, `test_poles_weight_eras_equally`,
  `test_missing_keyword_excluded_and_reported`, `test_emi_matches_paper_formula_on_toy_data`
  (hand-computed bins and z), `test_series_handles_empty_congress`, `test_layer_auc_separates_planted_axis`,
  `test_end_to_end_raw_scoring_on_tiny_bert`.
- [ ] Implement, run tests, commit.

### Task 4: TopK SAE and training (`sae.py`, `train_sae.py`)

**Files:** Create `src/emi_analysis/sae.py`, `src/emi_analysis/train_sae.py`, `tests/test_sae.py`.

**Interfaces — produces:**
- `TopKSAE(d_in, n_features, k)`: `set_normalization(mean, scale)`, `encode(x) -> Tensor (n, F)` (raw inputs),
  `decode(f) -> Tensor (raw space)`, `forward(x) -> (x_hat, f, info)`, `loss(x, dead_mask) -> (loss, parts)`
  with normalized MSE + 1/32 AuxK; `normalize_decoder()`; `save(path, meta)`; `load_sae(path, device) -> (sae, meta)`.
- `train_sae.find_layer_module(model, base, batch, layer) -> nn.Module` (module whose output equals
  `hidden_states[layer]`), used to splice reconstructions.
- CLI: `--model`, `--kind token|speech`, `--layer N | --layer-from results/emi/raw/<model>/<run>/meta.json`,
  `--activations DIR` (speech kind and token budgets), `--k 64`, `--expansion 8`, `--seed 0`,
  `--tokens 200000000` (token), `--samples 50000000` (speech), `--batch-size 4096`, `--lr 2e-4`,
  `--precision`, `--output-root sae`. Output `sae/<model>/<run>/step<N>/<kind>/layer<L>_k<k>_x<e>_seed<s>/`:
  `sae.pt`, `history.csv`, `metrics.json` (FVU, dead fraction, spliced LM loss delta for token kind, era
  token counts, ModernBERT top-5-dim variance share).

- [ ] Tests: `test_topk_keeps_exactly_k`, `test_encode_decode_roundtrip_improves_with_training`,
  `test_save_load_roundtrip`, `test_find_layer_module_matches_hidden_state` (bert, gpt2, qwen3, last layer
  included), `test_token_training_is_era_balanced_and_excludes_heldout`, `test_speech_training_end_to_end`.
- [ ] Implement, run tests, commit.

### Task 5: Token-SAE corpus pass (collector `--sae` mode) and verification

**Files:** Modify `src/emi_analysis/collect_activations.py`, `src/emi_analysis/verify_activations.py`;
test in `tests/test_collect_activations.py`, `tests/test_verify_activations.py`.

**Interfaces:** `collect_activations --sae PATH` pools SAE(token at the SAE's layer) per speech; meta gains
`layers_stored: [L]`, `sae: {path, layer, k, n_features}`, `representation: token_sae`; fingerprint includes
the SAE path; output tag defaults to `<tag>_tokensae_<sae dir name>`. Verifier reads `layers_stored`, passes
`--sae` to the recomputation, and for SAE collections checks row identity by token count + cosine ≥ 0.99.

- [ ] Tests: `test_sae_collection_equals_mean_of_token_features`, `test_sae_collection_refuses_other_sae_on_resume`,
  `test_verify_passes_sae_collection`.
- [ ] Implement, run tests, commit.

### Task 6: Ablations and validation drivers

**Files:** Create `src/emi_analysis/emi_ablations.py`, `src/emi_analysis/emi_validate.py`,
`tests/test_emi_ablations.py`.

**Interfaces:** `emi_ablations --model --representation --scores DIR (Task 3 output) …` writes
`results/emi/ablations/<representation>/<model>/{ablations.csv, era_pole_cosines.csv, loo_keywords.csv,
series_<variant>.csv}`; each ablation row: `ablation, variant, pearson_speech, spearman_speech,
pearson_series, n_scored`, against the primary held-out EMI. Implements ablations 2, 3, 4, 7, 8, 9, 11 within
one representation; 1, 5, 6, 10 by comparing score folders (`--compare DIR …`, `--base-activations`,
`--base-keywords`). `emi_validate`: `--export-excerpts` writes the Prolific excerpts as JSONL; then
correlates per-Congress series with the W2V series and excerpt EMI with mean(evidence_based − evidence_free).

- [ ] Tests: `test_ablation_table_has_every_variant`, `test_loo_drops_one_keyword_each`,
  `test_w2v_year_to_congress_mapping`, `test_excerpt_targets_average_raters`.
- [ ] Implement, run tests, commit.

### Task 7: SLURM scripts and README

**Files:** Create `slurm/keyword_occurrences.sh` (CPU, `short`), `slurm/keyword_activations.sh`,
`slurm/emi_score.sh` (CPU, `normal`, memory-heavy), `slurm/train_sae.sh`, `slurm/emi_ablations.sh`,
`slurm/emi_validate.sh`; README section "Contextual EMI" with the run order.

- [ ] `bash -n` every script; full test suite passes; commit.
