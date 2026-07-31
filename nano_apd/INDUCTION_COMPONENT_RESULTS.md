# Cross-layer components for controlled induction

The compact raw JSON reports are checked in under
[`results/induction_components`](results/induction_components/README.md).

## What was trained

The mechanism is **random-token induction**, not a semantic category such as apples or
potatoes.  A sequence contains an earlier random `cue -> label` pair and later repeats
the cue.  At the later cue, the desired output is the earlier label.  The matched
negative keeps the query and desired label fixed but replaces the earlier demonstrated
label with a distractor.  The score is the frozen model's positive-minus-negative
logit-margin difference.

This deliberately removes token meaning from the first experiment.  If a method cannot
recover a controlled copying mechanism, natural-language concepts will make the source
of failure harder to identify.

Each learned component is one object spanning all 96 editable linear maps in all 24
layers of `EleutherAI/pythia-410m`.  Its pieces are rank one in each matrix.  For module
`m`, the parameterization is

```text
W_m = residual_m + sum_c component_(m,c)
```

The residual is implicit.  Enabling every component exactly recovers the original
model; disabling every component leaves the residual.  There is no layer-locality or
head-locality penalty.  The sum-norm penalty first combines a component's squared mass
over every matrix and only then takes its norm, so moving the same component across
layers is not penalized.

Attribution is computed for the positive-minus-negative induction score, coherently
summed across all matrices, squared, and normalized so a row's gates sum to one.  The
training objective combines:

- routed output reconstruction and induction-gap reconstruction;
- removal of induction from the residual while preserving the matched negative;
- sparse per-example routing, dataset coverage, and use balance;
- the whole-component sum norm;
- random routed coalitions plus intact-model single-component ablations;
- individual and collective preservation on cue-corrupted controls; and
- natural-text preservation.

The current trainer additionally rotates single-component natural-text ablations.  This
last loss was added after the runs below: the results must not be read as evidence from
that new loss.

## Controlled variants

Rows are selected once using the frozen model's standard-induction behavior, then the
same rows are reused under every intervention.

Induction-present variants change lag and demonstration structure: standard, short lag,
long lag, two demonstrations, and a counterfactual label.  A recency-conflict variant is
a stress test.  Source-cue and destination-cue corruption are induction-absent controls.
The source detector checks both the repeated cue and the following positive/negative
label difference; this avoids mistaking an accidental random-token collision for the
demonstration.

## Results

The main functional audit uses 128 held-out, target-correct examples.  “Intact
necessary” means the 95% bootstrap confidence interval for the induction-margin damage
from removing that component from the otherwise untouched model is above zero.  It does
not mean the component passes the separate attribution-routed test.

| C | Steps | Intact necessary | Positive on all 5 present variants | Present effect > absent-control effect | Effective routed C | Routed gap / target gap | Examples owned | Routed necessity passes |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 8 | 300 | 8/8 | 8/8 | 8/8 | 3.72 | 8.77 / 10.81 | 7/8 | 3/8 |
| 16 | 350 | 16/16 | 16/16 | 16/16 | 5.07 | 9.83 / 10.81 | 12/16 | 0/16 |
| 32 | 400 | 32/32 | 32/32 | 32/32 | 5.19 | 9.02 / 10.81 | 9/32 | 1/32 |

The routed-necessity column uses a separate 256-pair evaluation and asks whether a
component is necessary on examples it owns after sum-normalized attribution routing.
That result is poor and should not be hidden.  The intact-model and routed claims answer
different questions:

- The intact result says every trained component has a reliable causal effect on the
  controlled task.
- The routed result says the attribution gate usually does not assign enough unique
  responsibility to all components.  Increasing C beyond 16 does not fix this; routing
  still uses about five effective components and ownership gets worse.

Thus the runs establish trainable distributed causal dependence, but not a clean
16-way explanation.  In particular, forcing every component to be necessary can split
one computation into interacting pieces without making those pieces independently
meaningful.

### Is one component absorbing everything?

For C=16, mean single-ablation damage over the five present variants ranges from 0.157
to 1.461 logit-margin units.  Component 15 is largest, but accounts for only 16% of the
sum of these non-additive single-ablation effects.  It is not the component with the
largest natural-text perturbation.  Its induction-present selected KL is 4.25 times its
natural-text KL, the best ratio in the C=16 run.  This argues against the simplest
“component 15 contains the whole model” explanation.

Component 7 is more ambiguous.  Its induction-margin damage is selective, but its
overall selected-distribution KL is almost unchanged between induction-present and the
synthetic induction-absent controls.  It may contain shared output machinery that helps
induction rather than a narrow induction-only operation.

Single-ablation effects are not additive, so the 16% number is a dominance diagnostic,
not a decomposition of total causal credit.

### Natural-text check

Each component was removed from the otherwise intact model at 128 random prediction
positions in the Pile `val` split.

| C | Median component KL on natural text | Maximum component KL | Median induction/natural KL ratio | Minimum ratio | Residual KL on natural text |
|---:|---:|---:|---:|---:|---:|
| 8 | 0.0198 | 0.0229 | 3.12 | 2.58 | 0.0669 |
| 16 | 0.0226 | 0.0339 | 2.99 | 2.25 | 0.1504 |
| 32 | 0.0170 | 0.0453 | 2.88 | 1.94 | 0.1470 |

All runs perturb induction more than an average natural position, but none is
functionally silent on natural text.  Average KL also cannot rule out a rare second
function.  This is evidence of task selectivity, not proof of natural-text
monosemanticity.

### Attribution plus direct head ablation

This was run last on C=16.  All 384 attention heads were tested on 128 held-out pairs,
of which 50 met the frozen model's target-correct positive-gap criterion.

- The top-32 attribution and top-32 direct-ablation sets overlap on 15 heads (Jaccard
  0.306).
- Fifteen of the top-32 attributed heads also have a positive causal confidence
  interval.
- The strongest convergent head is layer 5, head 2: removing it reduces the induction
  gap by 3.97 on average, with a 95% interval of [3.25, 4.70].
- Every learned component has at least one of the 15 convergent heads among its top eight
  attributed heads; the per-component overlap ranges from one to five.

This is partial circuit correspondence, not ground truth.  Some highly attributed heads
have negative or statistically unclear ablation effects, and some strongly causal heads
also change the matched negative substantially.  Attribution alone is therefore not a
safe causal label.

## What should happen next

Use C=16 as the iteration target.  C=8 is capacity-limited for a 16-part explanation;
C=32 creates more intact dependencies without producing finer attribution routing.

The next run should use the new rotating **individual** natural-text preservation loss,
then be accepted only if all of the following hold on untouched data:

1. All 16 intact single-ablation confidence intervals remain positive.
2. Every component owns enough examples under the routed model, and routed
   single-component ablations—not only intact ablations—are reliably positive.
3. Every component has a clear induction-versus-corrupt-cue and
   induction-versus-natural-text contrast.  Component 7 in the current run would fail
   this stronger criterion.
4. No component dominates the intervention effects, while routing still reconstructs
   most of the original induction gap.
5. Decoded natural-text top-effect examples are inspected for a repeated second
   function.  Mean KL alone is not enough.

If routed ownership remains near five effective components, do not increase C or add a
stronger “all components necessary” penalty.  That would manufacture more interacting
dependencies.  Instead, train functional specialization explicitly: build controlled
counterfactuals for source matching, label binding, distractor suppression, and
destination copying; require sparse component use within each case; and test that a
component's causal effect follows its proposed operation while leaving the other cases
alone.  Those roles should be treated as hypotheses and rejected when the interventions
do not separate them.

## Reproduction

The relevant entry points are:

```bash
torchrun --standalone --nproc_per_node=2 \
  -m nano_apd.train_induction_components --C 16 --bf16 --tag NAME

python -m nano_apd.eval_induction_roles ARTIFACT \
  --n_pairs 128 --batch_size 4 --component_chunk 4 --use_bf16

python -m nano_apd.eval_induction_natural ARTIFACT \
  --n_sequences 128 --batch_size 4 --component_chunk 4 --use_bf16

python -m nano_apd.eval_induction_components ARTIFACT \
  --n_pairs 128 --batch_size 8 --component_chunk 4 --use_bf16 \
  --ground_truth --head_chunk 4
```

The result JSON files include explicit warnings where a measurement supports only a
narrower claim than “monosemantic component” or “mechanistic ground truth.”
