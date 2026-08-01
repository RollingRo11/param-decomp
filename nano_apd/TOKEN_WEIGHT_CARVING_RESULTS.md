# Token-conditioned weight carving

## Question

Can one selected-token contrast identify a small, causal set of existing parameter
directions without training an APD dictionary?

This experiment is a deliberately cheap alternative to learned decomposition. For each
linear matrix in Pythia-410M, it forms the clean-minus-corrupt gradient of a fixed
target-versus-distractor logit margin, computes the leading four singular directions,
and projects the existing weight into four fixed rank-one pieces. There is no optimizer
and no learned residual: a physical gate subtracts a piece from the otherwise intact
model. Sum-normalized squared attribution is used only to rank pieces, never as the
physical gate value.

The experiment uses four induction prompt families for discovery, then disjoint screen
and confirmation sets. Screening reserves at least one candidate from every one of the
24 layers before selecting the final 16 by exact leave-one-out effect. The final
shortlist is consequently cross-layer, but each candidate is still a **module-local
rank-one direction**, not one APD-style component whose index spans all layers.

## Protocol

- Model: `EleutherAI/pythia-410m`, revision
  `9879c9b5f8bea9051dcb0e68dff21493d67e9d4f` (405,334,016 parameters).
- Candidate bank: rank 4 in the QKV, attention-output, MLP-input, and MLP-output
  matrices of every layer: 96 matrices and 384 rank-one candidates.
- Discovery: four eligible pairs from each of standard, short-lag, long-lag, and
  two-demonstration induction prompts.
- Selection: a disjoint 16-pair screen, 32 layer-covered candidates, then a frozen
  16-candidate shortlist.
- Confirmation: 32 disjoint eligible standard pairs, six induction-present variants,
  two cue-destroyed controls, 32 stochastic coalitions, and a separate unfiltered
  64-pair population.
- Readout: clean-minus-corrupt difference in a fixed target-token versus demonstrated-
  distractor margin at the selected position.
- Numerical check: 16-node Gauss-Legendre integration in FP32, plus exact endpoint
  ablations. Discovery runs in BF16.
- Geometry controls: ordinary Euclidean singular vectors and diagonal-KFAC-whitened
  singular vectors. Both use the paired projection.

The two geometry arms ran independently on the two H100s. DDP would add synchronization
without helping here because extraction is analytic and has zero optimizer steps.

## Results

### Cost and selection

| Metric | Euclidean | Diag-KFAC |
|---|---:|---:|
| Total time, including model load | 289.9 s | 288.3 s |
| Model load | 155.5 s | 153.4 s |
| Discovery and extraction | 6.3 s | 6.3 s |
| Peak CUDA memory | 5.22 GiB | 5.22 GiB |
| Optimizer steps | 0 | 0 |
| Effective attribution components, mean | 13.51 | 15.61 |
| Final layers / module paths | 11 / 15 | 11 / 15 |
| Supporting / suppressive / inconclusive | 9 / 7 / 0 | 9 / 7 / 0 |

The screen labels were frozen before confirmation. All 16 labels reproduced exactly on
the confirmation split, and each effect had the same direction on every one of its 32
prompts. The confidence intervals below are nominal per-candidate 95% bootstrap
intervals; they are unadjusted for testing 16 candidates.

The strongest Euclidean supporter was candidate 168, layer-10 MLP-input mode 0:
`+5.07 [4.36, 5.73]`. Its strongest suppressor was candidate 380, layer-23 MLP-output
mode 0: `-5.89 [-6.38, -5.39]`. Under diagonal-KFAC, candidate 168 remained the strongest
supporter at `+3.07 [2.66, 3.47]`, while candidate 112, layer-7 QKV mode 0, was the
strongest suppressor at `-3.17 [-3.84, -2.56]`.

### Joint causal importance

| Confirmation intervention | Euclidean | Diag-KFAC |
|---|---:|---:|
| Full contrast margin | 19.83 | 19.83 |
| Margin after removing the 9 supporters | 6.31 | 8.00 |
| Support-group damage | 13.52 | 11.83 |
| Fraction of the selected mean margin removed | **68.2%** | **59.7%** |
| Margin after removing the 7 suppressors | 37.54 | 31.82 |
| Suppressor-group endpoint | -17.71 | -11.99 |
| Margin damage after removing all 16 | 0.83 | 3.36 |

The support-only intervention is the clearest positive result. The nine frozen
supporters jointly carry 60--68% of this selected-token contrast on held-out prompts.
Removing suppressors improves the contrast, so removing all 16 together is a misleading
summary: opposing effects and interactions largely cancel. Individual effects are also
not additive; the Euclidean supporters sum to 18.58 individually but cause 13.52 damage
jointly (15.27 versus 11.83 under diagonal-KFAC).

This is necessity for a chosen logit contrast, not proof that every piece is required
for the expected token. Removing a single supporter changed the clean top prediction on
only 18.8% of prompts on average under Euclidean geometry and 14.2% under diagonal-KFAC.

### Integration and replication

- Support-group integration matched the exact endpoint with mean absolute completeness
  error 0.048 Euclidean and 0.045 diagonal-KFAC. Independent per-candidate integrated
  leave-one-out error averaged 0.0097 and 0.0085 respectively.
- On 64 separately generated, unfiltered pairs, all 16 effects kept their confirmation
  sign and all nominal intervals excluded zero. Mean absolute effect retention was
  90.4% and 88.8%. Clean target accuracy was only 42.2%, so this replicates the fixed
  contrast rather than successful induction behavior on every prompt.
- In sampled coalitions, all 16 marginal effects kept their exact leave-one-out sign;
  the worst per-candidate directional consistency was 98.7% and 99.1%.
- Mean absolute effect retention was 87--103% on short-lag, long-lag,
  two-demonstration, and counterfactual variants, and 61--65% under recent conflict.
  It fell to 12--14% when either induction cue was destroyed.
- The absent-control effects attenuated but usually remained statistically nonzero.
  These pieces are induction-enriched, not induction-exclusive.

### Geometry stability

The two shortlists shared 11 of 16 candidate IDs (Jaccard 0.524), with the same causal
sign for all 11 and Pearson correlation 0.961 between their exact effects. Supporting
module paths were more stable than individual singular modes: eight of ten supporting
module paths agreed between geometries (Jaccard 0.800). Fourteen of 16 Euclidean pieces
and 15 of 16 diagonal-KFAC pieces used local mode 0.

No attention-output candidate survived either selection. The final Euclidean set
contains 5 QKV, 6 MLP-input, and 5 MLP-output pieces; diagonal-KFAC contains 7, 5, and 4.
The selected pieces occupy only 0.000366% and 0.002718% of squared Frobenius weight mass,
although that small geometric mass should not be interpreted as a fraction of model
computation.

## Interpretability audit

The causal result does **not** currently support a claim of 16 distinct monosemantic
mechanisms.

- Exact behavioral fingerprints are extremely redundant: their median nearest-neighbor
  cosine is about 0.997, and their first principal component explains 98.5% and 98.4% of
  variance. They mainly encode supporter-versus-suppressor strength, rather than
  candidate-specific functions.
- Positional labels are weak enrichment labels derived from only four standard discovery
  pairs. Only 5/16 Euclidean and 3/16 diagonal-KFAC candidates put a majority of their
  attribution mass in the named region. They should not be called monosemantic labels.
- The matched direct-head reference tests all 384 attention heads on the same confirmation
  prompts and readout. Positive reference mass is concentrated at layers 5 (41.5%) and
  11 (26.7%), while carved supporters concentrate at layers 10 and 22.
- Cosine similarity between the supporting exact-effect layer profile and the matched
  positive-head profile is only 0.117 Euclidean and 0.133 diagonal-KFAC. Restricting
  carved supporters to attention modules reduces it to 0.000 and 0.008.

Thus the method finds a sparse, stable, task-linked mixture of supporting and suppressive
weight directions, but it does not recover the positive induction-head circuit. The most
plausible reading is that it selects downstream MLP machinery plus a few QKV directions,
with many directions exposing nearly the same input-output behavior.

## What is and is not supported

Supported by this run:

- Useful parameter directions can be carved from a 405M-parameter model in minutes,
  with no dictionary training and modest memory.
- A screen-frozen nine-piece support group carries a large fraction of one held-out
  selected-token contrast.
- Causal signs are stable across a second geometry, prompt variants, an unfiltered
  population, and stochastic coalition contexts.

Not supported by this run:

- Sixteen causally necessary expected-token mechanisms.
- Sixteen monosemantic or functionally distinct components.
- Recovery of the known positive induction-head circuit.
- APD-style cross-layer components: the bank is global only for ranking and gating;
  every carved direction belongs to one matrix.
- Broadly safe editing or unlearning. KL and token flips were measured only at the
  selected task position, not across unrelated text or the full model distribution.

The most principled next experiment is therefore not a larger candidate bank. It is a
second-stage diversity constraint or residualization over **behavioral effects**, followed
by the same frozen-split causal audit. Such a step earns its complexity only if it lowers
fingerprint redundancy while preserving support-group coverage and cross-geometry
stability.

## Reproduction

Run the two geometry controls concurrently, one per H100:

```bash
CUDA_VISIBLE_DEVICES=0 python -m nano_apd.run_token_weight_carving \
  --model_name EleutherAI/pythia-410m --rank 4 --geometry euclidean \
  --projection paired --bf16 --tag induction_r4_corrected

CUDA_VISIBLE_DEVICES=1 python -m nano_apd.run_token_weight_carving \
  --model_name EleutherAI/pythia-410m --rank 4 --geometry diag_kfac \
  --projection paired --bf16 --tag induction_r4_corrected
```

The runner refuses to overwrite an existing output directory unless `--overwrite` is
passed. Full raw outputs and reloadable `pieces.pt` checkpoints are written under
`nano_apd/out/` and intentionally ignored by git. Compact, per-example-stripped result
files are checked in under [`results/token_weight_carving`](results/token_weight_carving/).
