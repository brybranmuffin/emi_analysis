# Handoff: Untruncated Continued Pretraining + Verification

## Context

This repo implements a DDR/EMI pipeline over ~8M US House speeches (1879–2022). It
continued-pretrains BERT and GPT-2 on the corpus, trains sparse autoencoders on the
resulting activations, and computes Evidence-Minus-Intuition (EMI) scores.

**The bug we are fixing:** the training data loader truncated speeches (256 tokens for
BERT, 512 for GPT-2) instead of chunking them. Tokens past the cutoff never produced
gradient. Because congressional speeches open with procedural boilerplate ("Mr. Speaker,
I yield myself such time as I may consume"), the models were adapted mostly to speech
*preambles* rather than to argumentative content. The two models were also truncated at
different lengths, so they were adapted on non-equivalent corpora.

**This week's job:** rebuild the data pipeline so no tokens are discarded, re-run
continued pretraining on BERT and GPT-2, and verify both that adaptation worked and that
it did not break anything else.

## Scope

**In scope:** data pipeline, continued pretraining of BERT and GPT-2, verification.

**Out of scope — do not touch these this week:**
- SAE training (runs next week, after adapted models exist)
- Adding ModernBERT or Qwen3 (next week)
- Changing the EMI calculation, pole vector construction, or seed word lists
- Pooling / aggregation changes

Verification this week runs on **raw pooled activations**, not SAE features. This keeps
"did adaptation work" independent of SAE quality.

## Step 0 — Orient

Read the existing code and report back before writing anything:

1. Where does truncation happen? Find the tokenization / data loading code and confirm
   the max_length and truncation settings for each model.
2. How are the continued pretraining runs currently launched? Note the config format,
   hyperparameters, and checkpointing behavior.
3. Where is EMI computed, and what does it take as input?
4. Is there an existing train/test split? If so, how was it made?

Write a short summary of what you found. Do not start rebuilding until that is done.

## Step 1 — Held-out split (do this first)

Hold out a test set **before** any other data processing, so nothing leaks.

- Split at the **speech** level, not the chunk level. All chunks from one speech must
  land on the same side.
- Stratify by Congress, so the held-out set spans the full 1879–2022 range.
- Target ~20k speeches held out. Save the speech IDs to a file and keep it fixed.
- Also prepare a small general-domain held-out sample (~2k sequences from Wikipedia or
  OpenWebText) for the forgetting check in Step 5.

## Step 2 — Rebuild the data pipeline

Three changes:

**Chunking (the main fix).** Split each speech into consecutive non-overlapping windows
of the model's max length. A 1400-token speech at window 512 becomes 3 chunks (512, 512,
376), not 1 truncated chunk. No tokens are discarded. Drop trailing chunks shorter than
~32 tokens, since they are mostly noise.

**Packing.** To avoid wasting compute on padding, pack multiple short chunks into one
training sequence. If you implement packing, you must also mask attention at document
boundaries so tokens cannot attend across speeches.

> If boundary-masked packing turns out to be fiddly with the existing training code, use
> **length bucketing** instead: sort chunks by length and batch similar lengths together.
> It is less efficient but much simpler and correct by construction. Prefer the simple
> version that works over the clever version that might not.

**Deduplication.** Congressional speech is extremely repetitive. Run near-duplicate
detection (MinHash/LSH, or normalized-text hashing if that is simpler) and drop
near-identical speeches. Report how many were removed, broken down by era.

### Validate the pipeline on 10k speeches before the full run

Take a 10k-speech subset and run the whole thing end to end: chunk → pack → train a few
hundred steps → extract activations → compute EMI. Check specifically:

- Total tokens out ≈ total tokens in, minus dedup and minimum-length drops. Print both
  numbers. This is the main assertion that the truncation bug is gone.
- No chunk exceeds the model's max length.
- With packing on, verify attention masks actually block cross-speech attention (inspect
  one batch by hand).
- Held-out speech IDs appear nowhere in the training set.

Do not launch the full-corpus run until all four pass.

## Step 3 — Baseline EMI (before any training)

Compute EMI on the held-out speeches using the **off-the-shelf** (not yet adapted) BERT
and GPT-2. Forward passes only, no training. Save the per-speech scores.

This is the reference point for the final check in Step 6. It is cheap and it must exist
before training starts.

## Step 4 — Continued pretraining

Run BERT and GPT-2 separately on the full corpus with the rebuilt pipeline.

- BERT: masked language modeling, 2–3 epochs, fresh masks each epoch (MLM only produces
  loss on masked positions, so it needs more passes).
- GPT-2: causal language modeling, 1–2 epochs.
- Keep hyperparameters otherwise the same as the existing runs unless something is
  clearly wrong. We are isolating the effect of the data fix.
- **Checkpoint every 10% of training.** Run the Step 5 metrics at each checkpoint. This
  matters: it lets us catch problems a third of the way in instead of after a full run.

## Step 5 — Verification at each checkpoint

Three measurements, taken on the unadapted model first and then at every checkpoint.
Write all of them to one CSV: `model, step, congressional_loss, general_loss, mean_seed_drift`.

**1. Held-out congressional loss.** MLM loss for BERT, perplexity for GPT-2, on the
held-out speeches. Should fall. Compare each model only against its own starting value —
MLM loss and perplexity are not on a common scale, so never compare BERT's number to
GPT-2's.

**2. Held-out general-domain loss.** Same metric on the general-domain sample. Should
stay roughly flat. A sharp rise means the model is overfitting to congressional text and
losing general language ability.

**3. Seed-word drift.** For each evidence and intuition seed word, compute the cosine
distance between its embedding now and at the start of training. Report the mean and the
max. Should stay small. The seed words define the EMI axis, so if they move, we are
measuring along a different dimension than we started with.

**Flag immediately if, at any checkpoint:** general loss rises sharply, or mean seed
drift is large and still climbing. Either means stopping the run and reporting back —
the fix is to mix 10–20% general-domain text into training, but do not do that
preemptively.

## Step 6 — Final check: did adaptation change EMI?

Compute EMI on the held-out speeches with the adapted models, and correlate against the
Step 3 baseline (Pearson and Spearman, per model).

This is the measurement that decides whether continued pretraining stays in the pipeline
at all. Falling loss proves the model learned the corpus; it does **not** prove the EMI
measure improved. The two can come apart.

- Correlation ~0.95+ → adaptation is not changing the measure, and we may be able to drop
  it entirely.
- Lower → adaptation is doing real work and is justified.

Both outcomes are useful. Report the number plainly either way; do not try to make it
come out one way.

## Deliverables

1. Rebuilt pipeline code, with the 10k-speech validation checks runnable as a script.
2. Held-out speech ID file.
3. Adapted BERT and GPT-2 checkpoints.
4. `metrics.csv` with the per-checkpoint verification numbers.
5. Three plots: congressional loss vs. step, general loss vs. step, mean seed drift vs.
   step (all with both models on the same axes where scales permit).
6. A short `RESULTS.md`: token counts before/after the fix, dedup counts, the final EMI
   correlations, and anything that looked wrong.

## Notes

- Keep the old truncating loader intact behind a flag. We need to be able to reproduce
  the original results for comparison.
- Print token counts loudly at every stage. The whole point of this week is that no
  tokens go missing, and we want that visible, not buried in a log.
- If something in the existing code is ambiguous or looks broken, ask rather than
  guessing. Do not silently refactor things outside this scope.
