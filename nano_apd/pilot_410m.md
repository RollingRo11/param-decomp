# Pythia-410M engineering pilot (20 updates)

This is an engineering check, not evidence for a scientific claim. It used one training
seed, only 20 optimizer updates, synthetic matched-induction pairs, and a 64-sequence
retain pool. The purpose was to exercise the two-H100 implementation and reject obviously
bad formulations before spending on a real sweep.

## Setup

- Target: `EleutherAI/pythia-410m` (405,334,016 parameters), frozen.
- Runtime: two H100 80GB GPUs, one process per GPU, BF16 forwards, averaged bank
  gradients.
- Data: length 64; global matched batch 16; global retain batch 8. Positive and negative
  contexts differ at one earlier token. By default only examples where the target
  predicts the selected positive token and has positive contrast margin are supervised.
- Localization: the four highest contrast-usage matrices from 16 calibration pairs.
- Arms: eight rank-2 free pieces; one rank-16 free piece; eight rank-2 projected pieces.
- Held-out evaluation: 64 fresh pairs (18 eligible in BF16 and 16 in FP32), 32 fixed
  natural Pile sequences, eight random component subsets.

## Results

BF16, at each free arm's full edit strength:

| arm | eligible gap removed | eligible negative KL | retain KL | subset Pearson | subset MAE |
|---|---:|---:|---:|---:|---:|
| 8 x rank 2 free | 6.411 | 0.620 | 0.0305 | 0.981 | 0.265 |
| 1 x rank 16 free | 5.356 | 0.495 | 0.0176 | not meaningful | 0 |
| 8 x rank 2 projected | 0.027 | 0.048 | 0.0123 | -0.071 | 0.162 |

The free-bank strengths are unequal, so the dose curve is the fairer comparison. At
roughly matched efficacy, the eight-piece arm at 1.0x removes 6.411 margin points with
retain KL 0.0305. The rank-16 arm at 1.25x removes 6.239 with retain KL 0.0201. Eligible
negative KL is 0.620 versus 0.644. The overcomplete bank therefore does not dominate the
single low-rank edit: it slightly improves the matched-negative number at this operating
point but damages broad retain text more.

The eight free pieces do show a real compositional signal: their unseen-subset effects
have Pearson 0.981 and MAE 0.265, and their individual effects sum to 7.714 versus a full
effect of 6.411 (ratio 0.831). That is substantially better than the earlier unfiltered
pilot, whose subset MAE was 2.736. It is still not enough to call the pieces distinct
mechanisms: all eight individual effects point in broadly the same direction, and the
rank-matched edit has the better retention frontier.

The projected control failed to learn a useful edit. Its original scale initialization
was also saturated; after fixing that implementation error, the 20-update FP32 effect was
still only 0.052 margin points. BF16 produces a larger, non-monotonic response than FP32,
which is a warning that tiny projected edits are near quantization noise. Both precisions
are now saved separately by the evaluator.

Internal contrast changes are modest for the eight-piece bank: the two later selected
modules retain 95-96% of their contrast norm with cosine around 0.99. The rank-16 edit
changes those representations more (about 92% norm ratio and 0.98 cosine) while causing
less broad retain KL. This does not support stronger mechanistic removal by the
overcomplete arm.

## Decision

Do not scale this exact configuration or describe it as a decomposition. The next useful
experiment is a seed sweep over retention weight and component-budget weight, always
plotting the dose-matched efficacy/retention frontier against rank 16. Continue only if
the multi-piece arm wins that frontier across seeds and its component effects become
less redundant. The projected arm needs a longer optimization check, but its current
failure should remain visible rather than being treated as positive extraction evidence.
