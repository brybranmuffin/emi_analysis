# Contextual EMI: raw activations and two SAE versions — design

Date: 2026-11-04. Status: draft for approval.

## Intent

Re-measure Aroyehun et al.'s evidence-minus-intuition (EMI) score for every congressional speech using the
representations of our four continued-pretrained models (BERT, GPT-2, ModernBERT, Qwen3), instead of
Word2Vec. The construct definition stays identical to the paper: the same 49 evidence and 35 intuition
dictionary keywords, the same DDR-style averaging, the same scoring arithmetic. Only the representation
changes, so any difference from the Word2Vec series is attributable to context, not to a different
dictionary or a different scoring rule.

Three representations are built and compared:

1. **Raw** — the model's hidden states (all layers).
2. **Token-SAE** — features of a sparse autoencoder trained on individual token activations.
3. **Speech-SAE** — features of a sparse autoencoder trained on whole-speech average activations.

Everything SAE-related is trained fresh. Nothing (weights, prototypes, exemplar lists, scores, chosen
hyperparameters) is reused from the earlier `text-generation-project` repo; it is only background reading.

## Constraints from the professor (non-negotiable)

- **Poles never come from Word2Vec scores** (no "top-N speeches by W2V EMI" exemplars). Otherwise the new
  measure inherits what it is tested against.
- **Poles never come from what we validate against** (no LLM-picked or human-rated exemplars).
- **Poles are era-balanced**, so "close to the evidence pole" does not partly mean "close to the decades the
  pole came from."
- Poles are built from **the dictionary keywords as they occur in the speeches**: sample occurrences per
  keyword per era, take the model's activation at each occurrence, average within keyword, then across
  keywords (DDR), then center.
- **Drift test**: one set of poles per era, compared with each other and with the pooled poles (the
  contextual counterpart of Aroyehun SI S11).

## Goals

- Per-speech EMI for all 8,373,445 speeches, per model, for each representation (raw: every layer;
  SAE: the chosen layer).
- Per-Congress EMI series with bootstrap CIs, in the same shape as `congress_EMI_w2v_bootstrap_CIs.csv`.
- The ablation table (below), computed on the 20,000 held-out speeches.
- Validation against the paper's Word2Vec series and the Prolific human ratings, run last.
- Every run logged and resumable in the style of the existing pipeline (banner, progress, `meta.json`,
  SLURM scripts with mail, account `p53299`).

## Non-goals

- LLM-label validation (no labels exist yet; leave a slot for it).
- Interpreting individual SAE features.
- The paper's downstream regressions (polarization, inequality, productivity).
- Re-collecting the per-speech raw activations — the four verified collections are inputs.

## Anti-goals (a technically working result that would still be a failure)

- Choosing a layer, an SAE configuration, or any scoring option by looking at agreement with the W2V
  series, human ratings, or any other validation target.
- Poles that differ between models in which occurrences they saw (models must be compared on identical
  keyword occurrences).
- Reporting only the best model/layer/variant.

## Inputs

- Speech activations (verified, job 7129919):
  - `activations/bert/run-job4757959/step350847/corpus` (13 layers × 768)
  - `activations/gpt2/run-job9843505/step759917/corpus` (13 × 768)
  - `activations/modernbert/run-job4757963/step316779/corpus` (23 × 768, collected in strict fp32)
  - `activations/qwen3/run-job4757993/step481486/corpus` (29 × 1024)

  Rows are in `speech_ids.txt` order: `data/train.jsonl` then `data/heldout/congressional.jsonl` (last 20,000
  rows). `congress.npy` and `token_counts.npy` sit beside them. Row order is identical across models.
- Speech text: `data/train.jsonl`, `data/heldout/congressional.jsonl` (`speech_id`, `congress`, `text`).
- Lexicons: `data/lexicons/evidence_lexicon.csv`, `data/lexicons/intuition_lexicon.csv` (read with the
  existing `train.load_lexicon`). Counts are logged; if they are not 49/35 the log says so (the old notebook
  had 49/32) and the CSVs win.
- Validation: the EMI repo's `data/congress_EMI_w2v_bootstrap_CIs.csv` (per year) and
  `data/prolific_ratings_US_congress_congressw2v.csv` (ratings of ~150-word excerpts, with `speech_id`,
  `text`, `evidence_based`, `evidence_free`, `starting_year`).

## Decided choices

### Eras
Eight eras of nine Congresses: 46–54, 55–63, 64–72, 73–81, 82–90, 91–99, 100–108, 109–117
(1879–2022, 18 years each).

### Keyword occurrences (shared by all models and representations)
- Match each lexicon entry as an exact, case-insensitive whole word in the speech text (no stemming; the
  lexicon already lists the forms it wants, e.g. `findings`, `analyse`).
- Sample from **training speeches only** (not the 20k held-out), so held-out ablations and validation are
  uncontaminated.
- Per keyword per era: **100 occurrences**, at most one per speech per keyword, fixed seed. If an era has
  fewer, take all and log it.
- Sampling happens once, on character spans, and is saved to a file. Every model then maps the same spans to
  its own tokens. This is what makes the models comparable.
- A word that occurs in fewer than 20 sampled contexts overall is flagged in the log and report (kept, not
  dropped).

### Keyword activations
- For each occurrence, run the chunk that contains it, built exactly as the speech collection built it
  (same `chunk_token_ids` window, same `[CLS]/[SEP]` wrapping), so the keyword's state is computed in the same
  context as the speech vector it will be compared with.
- The keyword's vector is the mean over the tokens that overlap its character span (multi-piece words are
  averaged), at every layer.
- Also record the **keyword alone** (each lexicon word run by itself, with the model's usual wrapping) for the
  context ablation.
- Done for the adapted checkpoint (main) and the base model (ablation). ModernBERT runs in strict fp32;
  the others in bf16 (verified equivalent).

### Poles (same rule in every representation)
1. Map each occurrence's vector into the representation (raw: as is; SAE: encode with that SAE).
2. Keyword vector = mean over eras of (mean over that era's occurrences). Each era counts equally even when
   an era is short; an era with no occurrences is skipped and logged.
3. Pole = mean of its keyword vectors (DDR).
4. Center: subtract μ, the corpus **token-weighted** mean in that representation. Because the stored speech
   vectors are token-weighted means, the token-weighted mean over speeches equals the mean over all tokens.
   One μ therefore centers single-token poles and speech averages alike. μ is computed from the speeches
   (`Σ count_i·x_i / Σ count_i`). For the speech-SAE, μ is the token-weighted mean of the encoded speech
   vectors.

### Scoring (identical to `compute_sbert_avg_lexicon.py` except for the representation)
- Evidence score = cos(x − μ, P_ev); intuition score = cos(x − μ, P_int), with P the centered poles.
- Speeches with ≤ 10 words, or with < 5% of words from the 100 most common English words (the paper's
  `wordfreq` filter), get no score.
- Length adjustment: subtract the mean score within 10-word length bins (word counts from the text).
- z-score each adjusted score over all scored speeches, then EMI = z_ev − z_int.
- Per-Congress series: mean EMI per Congress, with bootstrap 95% CIs (1,000 resamples of speeches).
- Known difference from the paper: they scored 150-word chunks; we score whole speeches. Ablation 11 checks
  whether weighting speeches by their chunk count changes the series.

### Choosing the layer (fixed before any validation)
For each model and layer, measure how well the poles separate evidence from intuition **occurrences**:
leave one keyword out, build poles from the rest, and score that keyword's held-out occurrences. Report
AUC over all occurrences. The primary layer per model is the one with the highest AUC (ties → deeper).
Every layer is still scored and reported. This uses only the dictionary, never the validation data.

### Token-SAE
- Architecture: TopK SAE (Gao et al., 2024), the encoder keeps exactly k features. Unit-norm decoder columns.
  Pre-encoder bias initialized to the data mean, inputs scaled by one scalar to unit mean squared norm.
  AuxK loss for dead features.
- Primary config: **k = 64, width 8×** (6,144 features for 768-dim models, 8,192 for Qwen3). This is a
  standard default chosen in advance, not tuned on EMI. The sweep is an ablation.
- Layer: the primary layer from the rule above (so raw scoring of that model runs first).
- Training data: about 200M tokens streamed from **training** speeches (held-out excluded) through the
  adapted model at that layer, special tokens excluded, shuffled through a buffer. Speeches are sampled so
  each era contributes the same number of tokens (my call: keeps the SAE's dictionary from being dominated by
  the eras with the most speech, in the same spirit as the era-balanced poles). Activations go straight
  from the forward pass into SAE training; nothing token-level is stored.
- Seeds: two. Seed 0 is used for the full corpus, and seed 1 is scored on the held-out 20k for stability.
- Diagnostics logged: fraction of variance unexplained, dead-feature fraction, and the change in held-out LM
  loss when the layer's activations are replaced by their SAE reconstruction. For ModernBERT, also log the
  share of variance in its top 5 dimensions (its very large outlier dimensions could dominate the
  reconstruction loss).
- Speech features: a new full pass of the corpus with the SAE attached at that layer, pooling SAE(token) per
  speech with the same token-weighted rule. This extends `collect_activations.py` (same chunking, packing,
  resume, verification) rather than a new collector. Stored as float16, about 100 GB per 768-dim model and
  about 140 GB for Qwen3. `verify_activations.py` is extended to check these collections.

### Speech-SAE
- Same architecture, config and seeds as the token-SAE, trained on the stored speech vectors of the same
  layer, training speeches only (8.35M rows), sampled so each era contributes equally (my call, as above).
- Speech features are computed on the fly from the stored vectors at scoring time, so nothing new is stored.
- Poles: keyword occurrence vectors (single tokens) are encoded with this SAE. That mismatch, an SAE that
  learned from averages applied to single tokens, is part of what this version tests.

## Ablations (on the 20k held-out speeches; main results on all 8.37M)

| # | Ablation | Question answered |
|---|---|---|
| 1 | Representation: raw vs token-SAE vs speech-SAE | Does sparse coding change the measure? |
| 2 | Keyword in context vs keyword alone | Is context what changes EMI vs Word2Vec? |
| 3 | Per-era poles: cosine between era poles and with the pooled poles; trend re-scored with each era's poles | Drift (SI S11 counterpart) |
| 4 | Era-balanced vs unbalanced occurrence sampling (unbalanced: the same 800 occurrences per keyword drawn uniformly from all training speeches) | How much era confounding balancing removes |
| 5 | Base vs adapted model, raw representation (Pearson/Spearman of speech EMI, as the handoff specifies) | Did continued pretraining change the measure? Needs a base-model collection of the 20k (cheap). |
| 6 | All models × all layers (raw) | Architecture and depth dependence |
| 7 | Leave one keyword out (trend with each keyword dropped) | Does one keyword drive the result? |
| 8 | Two cosines vs projection on (P_ev − P_int); length adjustment on/off; centering on/off | Sensitivity to scoring details |
| 9 | Occurrences per keyword-era: 25, 50, 100, 200 | Are the poles converged? |
| 10 | SAE seed 0 vs seed 1; SAE sweep k ∈ {32, 64, 128} × width {8×, 16×} | SAE stability and config dependence |
| 11 | Speech-weighted vs chunk-weighted per-Congress means | Does whole-speech scoring vs the paper's 150-word chunks matter? |

The held-out set never contributes keyword occurrences (see above), so the paper's "exclude the pole-source
speeches" check holds by construction for every held-out ablation.

## Validation (run last, nothing upstream depends on it)

- **Trend**: Pearson/Spearman between our per-Congress series and the paper's W2V series (years mapped to
  Congresses), for every model, layer and representation.
- **Human ratings**: the rated items are ~150-word excerpts, so the excerpts themselves are scored. They are
  written to a JSONL and run through `collect_activations.py` (raw) and the token-SAE collector, then scored
  with the same poles and μ. Correlate with `evidence_based − evidence_free` per item (mean over raters),
  alongside the W2V score in the file as a reference.

## Pipeline order and rough cost

1. `keyword_occurrences` (CPU, once): scan text, sample spans, word counts and top-100 fractions per speech.
   About 1 hour.
2. `keyword_activations` (GPU, per model, adapted and base): minutes each.
3. `emi_score --representation raw` (CPU, per model): μ, poles, layer AUCs, scores for all layers, series.
   I/O-bound read of 167–497 GB, probably 1–2 hours per model.
4. Base-model held-out collection (GPU, existing collector with `--base`): minutes each.
5. `train_sae --kind token` and `--kind speech` (GPU, per model): under 1 hour each per seed; the sweep is
   extra.
6. Token-SAE corpus pass (GPU, per model): about the same time as the original collections (2–6.5 hours).
7. `emi_score` for both SAE representations.
8. `emi_ablations` (held-out) and `emi_validate`.

Each step gets a SLURM script and tests on tiny models, in the existing style.

## Other calls made in writing this (check these)

- The top-100-words filter uses the `wordfreq` package, as the paper did (new dependency).
- The layer-selection AUC scores each held-out occurrence by cos(v − μ, P_ev) − cos(v − μ, P_int).
- Prolific excerpts get a Congress from `starting_year` with the paper's formula, `(year − 1789) / 2 + 1`.
- The base-vs-adapted ablation uses raw activations only, not the SAE representations.

## Left to the builder

- Module boundaries and file formats beyond what is stated (per-speech scores as `.npy` per model and
  representation, series and ablation tables as CSV, are suggested).
- How a keyword span that straddles a chunk boundary is handled (expected to be rare; log the count).
- SAE optimizer, learning-rate schedule, batch size and buffer size, within the stated design.
- Mapping the W2V series' years to Congresses.
