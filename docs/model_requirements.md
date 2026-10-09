# Model training requirements and verification rules

## Speech truncation fix

The original bug truncated speeches before the model could consume the full argument content:

- BERT max length: 256 tokens
- GPT-2 max length: 512 tokens

The fixed behavior is:

1. Split each speech into consecutive, non-overlapping chunks of the model's max length.
2. Keep every chunk; do not discard the remainder unless it is shorter than the minimum useful chunk size.
3. Drop trailing chunks shorter than roughly 32 tokens to remove mostly-noisy fragments.
4. For packing, concatenate short chunks within a batch only when the attention mask explicitly blocks cross-speech attention.
5. Use a held-out speech split created before any processing so no speech leaks across train/test boundaries.

## Required metrics at each checkpoint

Each checkpoint writes one row to a combined CSV with columns:

- `model`
- `step`
- `congressional_loss`
- `general_loss`
- `mean_seed_drift`

### BERT

- Training objective: masked language modeling (MLM)
- Congressional loss: MLM loss on held-out congressional speeches
- General-domain loss: MLM loss on a general-domain held-out sample
- Recommended checkpoint cadence: every 10% of total training steps

### GPT-2

- Training objective: causal language modeling
- Congressional loss: perplexity on held-out congressional speeches
- General-domain loss: perplexity on the general-domain sample
- Recommended checkpoint cadence: every 10% of total training steps

## Stopping conditions

Stop and report immediately when either condition is observed at a checkpoint:

- general-domain loss rises sharply relative to the starting model
- mean seed drift is large and still climbing

These indicate either overfitting to congressional text or a drift in the EMI-relevant embedding direction.

## Final EMI check

Before training, compute EMI on held-out speeches with the off-the-shelf BERT and GPT-2 models. After continued pretraining, recompute scores on the same held-out speeches and correlate them against the baseline:

- Pearson correlation
- Spearman correlation

Interpretation:

- ~0.95+: adaptation is not changing the EMI measure materially
- lower: adaptation is doing real work and is justified

The final result should be reported plainly, not forced to fit an expected trend.
