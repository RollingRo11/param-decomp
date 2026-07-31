# Contrastive selected-token parameter carving

## Claim

Given several contexts in which one behavior helps predict a chosen token, plus matched
contexts with the same chosen token where the behavior is absent, recover a small bank of
low-rank weight pieces that:

1. aligns with how the unedited model uses its parameters for the positive-vs-negative
   contrast;
2. can be removed to eliminate that contrast without changing the matched controls or
   broad retain data;
3. has predictable effects when pieces are removed individually or in unseen subsets.

This is deliberately not a claim to interpret every weight. For matrix `m`, the learned
pieces `P[m,k]` and implicit residual

```
R[m] = W[m] - sum_k P[m,k]
```

sum exactly to the target weight, but exact summation alone is trivial. Original-model
usage alignment and causal subset tests are what distinguish carving from an arbitrary
low-rank counter-program.

## Supervision

Each example selects one logit position and one label. By default, the scalar target is
the correct-token logit minus the largest competing logit; this avoids nearly zero
gradients when a model is already confident. Target-logit and log-probability scores are
also available as explicit controls. One backward over a batch gives every example's
gradient because sequences do not share a computation graph.

For a selected module, the example-level weight usage is

```
d score / dW = sum_t outer(pre[t], grad_post[t])
usage = (d score / dW) * W
```

where the sum includes every source position that affects the chosen prediction. This is
important for attention: reading only the destination position would omit the earlier
positions used to retrieve information.

The trainer uses two cheap summaries without materializing a full parameter-sized vector:

- **Total signed credit:** inner product of the weight gradient with the original weight.
  The sum of component credits is trained to match the positive-minus-negative target
  credit.
- **Random-coordinate sketch:** fixed coordinates from `(d score / dW) * W`. The sum
  of the learned pieces must reconstruct those sampled target-usage coordinates directly,
  and pairwise geometry between examples in component-credit space must match geometry in
  the target sketch.

The built-in induction pair differs at exactly one earlier token. Both contexts have the
same selected cue and next-token label, so token identity alone cannot solve the task.
Only pairs where the target actually predicts that positive label and has a positive
contrast gap are supervised by default; otherwise the procedure would learn to remove a
token association the target does not successfully express. Custom pairs are the
intended interface for bracket closing, factual lookup, or other behaviors.

## Banks

Two parameterizations are required controls:

- `free`: `P = A B`. This is expressive and cheap, but its full sum is algebraically a
  conventional rank-`C*r` update.
- `projected`: `P = q_i q_i^T W q_o q_o^T`, with learned orthonormal input/output bases and
  a nonnegative scale initialized to `1/C`. It can only select a two-sided projection of
  the existing weight. Failure here alongside free-bank success is evidence for editing,
  not extraction; a near-zero saturated scale is not an acceptable control.

Only modules with high positive-vs-negative target-use contrast are selected by default.
The residual is never materialized during training.

## Scaling and distributed training

The default real-model target is Pythia-410M. Under two-process DDP, each H100 holds a
frozen copy of the target and sees a different shard of matched and retain examples.
Only the much smaller carving-bank gradients are averaged. Checkpoints, logs, and W&B
writes are rank-0-only. This removes the cost of learning a whole-model dictionary, but
it does not make the target forward/backward free: selected-token usage still requires
two activation-gradient passes for each positive/negative batch.

## Causal losses

- Full removal must reduce the positive-minus-negative selected-token log-probability gap
  to a bounded floor.
- KL preserves all matched-negative and broad retain predictions, as well as every
  positive-sequence position except the selected prediction.
- Random component subsets are assigned an expected partial effect based on how much
  original-model component credit they remove. Their effect is trained toward that target,
  and matched-negative outputs remain under KL.
- A two-sided component-count budget is optional. It is a budget rather than raw sparsity
  minimization because earlier experiments showed that one-sided sparsity creates coarse
  catch-all components.

## Required comparisons and decision rules

Every `C` by rank-`r` run must be compared with one rank-`C*r` component using identical
modules, data, and optimization budget. Report all of:

- positive behavior removal and matched-negative KL;
- natural retain CE, split into behavior-like and other positions;
- every individual component's effect;
- actual versus sum-of-individual effects for unseen subsets;
- seed-to-seed subspace and causal agreement;
- the weight-projected arm;
- a held-out relearning attack and an internal representation/circuit metric.

Call the result a decomposition only if multiple components remain individually causal,
stable, and compositional. If only their sum matters, report it as an efficient low-rank
edit. Call it unlearning only when it generalizes to natural and adversarial contexts,
changes the relevant internal mechanism, and resists held-out relearning.

## Experiment ladder

1. The checked-in ground-truth causal sequence toy (`run_carving_toy.py`), which plants
   two known rank-one mechanisms, runs five seeds, and automatically compares the
   overcomplete bank with one rank-matched component. Then port the same test to TMDR.
2. Bracket closing on pile-4L as the positive control, using same-token matched contexts.
3. Natural induction as a hard case. Repeated random-token sequences remain a diagnostic,
   not the main efficacy distribution.
4. Run the same three arms directly on Pythia-410M using both H100s: an overcomplete
   free bank, one rank-matched free component, and the projected bank. Scale beyond 410M
   only if multiple components beat the rank-matched edit and pass subset and relearning
   tests across seeds.
