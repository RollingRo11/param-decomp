# Results

All headline numbers, as comparisons against the published baseline method (**VPD**: rank-1
per-matrix pieces, one independent gate each — trained by us on identical targets with matched
budgets unless noted). **Ours** = whole-network components (one rank-1 piece in *every* decomposed
matrix, one shared gate), trained as described in `method.md`.

## Terminology in one line each

- **CE-recovered** — keep only the components the gate marks important, delete the rest: what share
  of the model's task performance survives (100% = perfect; measured against deleting everything).
- **KL (ci-masked)** — same test, measured as divergence between output distributions (0 = perfect).
- **KL (all-on)** — sanity check with *nothing* deleted; must be ~0 or the decomposition doesn't
  even sum back to the model, and every other number is meaningless.
- **Adversarial KL** — an attacker picks the worst combination of "unimportant" components to
  delete; the damage they achieve (lower = more robust decomposition).
- **L0** — how many components are active per input (or per token), on average.
- **MMCS** — for each true mechanism, how well does the best-matching learned component align
  (1.0 = perfectly)?
- **Separation** — does each input feature get its own dedicated component? (1.0 = one each)
- **Coverage** — what fraction of features get any confident component at all? (Always read
  separation *with* coverage: dropping most features fakes perfect separation.)
- **Cross-layer** — fraction of components with substantial weight in more than one layer, i.e.
  mechanisms captured as one unit rather than split per layer.
- **Keep-only error** — delete everything *except* a feature's component: how badly does that
  feature's computation break? (lower = the component really does the job; "do nothing" ≈ 0.6)
- **L1 ratio** — total absolute value of all component entries ÷ same total for the original
  weights; 1.0 = components claim fully separate pieces of the weights.

## 1. TMS (toy model of superposition: 5 features stored in 2 dimensions)

One decomposed matrix; ground truth = one rank-1 mechanism per feature.

| | MMCS (1.0 = perfect recovery) |
|---|---|
| **Ours** (C=20, 10k steps) | **0.982** |
| VPD (same target, matched setup) | 0.927 |
| APD paper's own machinery (their reported number) | 0.998 |

Ours also reaches reconstruction error 0.003 with ~0.9 components active per input (correct: ~1
active feature → ~1 active mechanism). Single seed.

## 2. Toy model of cross-layer distributed representations (2-layer residual MLP, 100 features)

Each feature's computation is deliberately spread across both layers; a perfect decomposition has
one component per feature spanning both layers. VPD's per-matrix pieces cannot express this
directly, so the published recipe adds a clustering step afterwards (their Appendix A.8 method,
knob α = how finely to split).

**Mechanism recovery:**

| method | separation | coverage | cross-layer | keep-only error ↓ |
|---|---|---|---|---|
| VPD + clustering, α=0.01 | 0.01 | — | 1.00 | — |
| VPD + clustering, α=1 | 0.66 | — | 0.46 | 0.22 |
| VPD + clustering, α=10 | 0.90 | — | **0.06** | 0.37 |
| **Ours, rank-1 (3 seeds)** | **0.89–0.92** | **0.97–1.00** | **1.00** | **0.10–0.12** |

The baseline has a wall: its clustering knob buys separation only by destroying cross-layer
structure (0.90 separation forces 0.06 cross-layer). Ours gets both at once, with no clustering
step and no granularity tuning — the cross-layer grouping is learned during training.

**Reconstruction head-to-head** (same target, matched training; "MLP variance recovered" = how much
of the network's actual computation, beyond the residual passthrough, the gated components
reproduce):

| | reconstruction error ↓ | MLP variance recovered | active per input (L0) |
|---|---|---|---|
| **Ours** (full-rank variant) | **0.046** | ~87% | 0.65 |
| VPD (canonical, 160 pieces) | 0.194 | ~45% | 1.05 |

Note: a healthy decomposition of this model sits at L1 ratio ~8 — its mechanisms genuinely overlap
in the weights (superposition). Forcing the ratio toward 1 here *destroys* recovery; this is the
model that taught us the L1 must be dosed per target.

## 2b. Variable-rank components (same distributed-reps toy, rank cap R=8)

Components may use up to 8 rank-1 pieces per matrix (one shared gate per component, unchanged);
training pressure decides how much each uses. Ground truth is rank **1** per matrix, so the test is
whether the method *chooses* rank 1 and keeps one feature per component. "True rank" = SVD of the
materialized component (the piece count overestimates — pieces can mix and cancel); entries marked
~ are from the piece-based proxy (run predates the SVD metric).

| variant | separation | coverage | recon ↓ | true rank (mean / median) |
|---|---|---|---|---|
| rank-1 reference (structural) | 0.89–0.92 | ~1.0 | — | 1 (fixed) |
| no penalty (cap only) | 0.68 | 0.99 | 0.095 | ~at cap |
| Frobenius / nuclear-norm penalty | 0.17–0.61 | 0.89–0.99 | 0.071–0.092 | inert or merging |
| rank-count trim (strong dose) | 0.64 | 0.96 | 0.090 | ~3.4 |
| nested ranks, 4x budget | 0.91 | 0.99 | **0.085** | 3.2 / 2.8 |
| **nested + trim, 2x budget** | 0.90 | 0.93 | 0.108 | **2.6 / 2.0** |

Readings: rank freedom with no counter-pressure lets components pack several mechanisms into their
budget (separation 0.68). The Frobenius penalty makes that *worse* (its optimum is few fat merged
components) — rejected. The rank-count trim controls capacity but not identity. **Nested ranks**
(train random rank-prefixes so pieces become importance-ordered) is what restores
one-mechanism-per-component at full coverage while beating the cap-only control on reconstruction;
adding the trim then pushes ranks toward the ground truth (median 2 vs answer 1, 80/130 components
at rank ≤ 2) at half nested's budget. Single seed each.

**Dense-target counterpoint (2-layer induction toy, the Christensen & Riggs testbed).** On a model
where every position runs most of the network, every rank scheme — including rank-1 — converges to
interchangeable components (~1/24 of every matrix each, true rank ~13/16, no dedicated induction
crew; deleting the most copy-selective components hurts no more than deleting random ones), even
after raising minimality until gates became position-selective. Rank pressure cannot create
component identity on dense targets; role-based forces (interaction, entrywise L1) are the open
lever there.

## 3. Pythia-14M (real 6-layer language model, Pile-trained, MLPs included)

All 24 weight matrices decomposed. VPD baseline: 6,912 pieces, 20k training steps. Ours: C=4096
components (= 98k rank-1 pieces; ~2× VPD's per-step cost).

**Main comparison (matched 20k-step budget):**

| | VPD | **Ours** (IMP=3e-3, L1=3e-4) |
|---|---|---|
| CE-recovered | 82.2% | **82.6%** (best checkpoint ~84.5%) |
| KL, ci-masked ↓ | 1.46 | **1.39** (best 1.25) |
| KL, all-on (sanity) ↓ | 0.014 | **0.010** |
| L0 (share of pieces active/token) | 50 / 6912 = 0.72% | **22 / 4096 = 0.54%** |
| adversarial KL ↓ | **3.6** | 35.4 |

Replicated across two seeds nearly digit-for-digit (CE 82.6/82.6, L0 22.3/22.4, adversarial
35.4/35.6). **Reading:** we match the published method's sparsity and beat its reconstruction and
summation tightness; worst-case robustness is our one clear deficit (~10×), and it is structural —
unchanged by removing the L1, by 5× more training, or across seeds.

**Sparsity–reconstruction frontier (ours, C=4096):**

| minimality strength | L0 | CE-recovered |
|---|---|---|
| IMP=3e-3 | 0.54% | 82.6% |
| IMP=1e-2 | 0.16% | 75.7% |

**Does more training help?** (both GPUs, 5× steps, 2× batch):

| budget | best KL ci-masked ↓ | CE-recovered |
|---|---|---|
| 20k steps | 1.25 | ~84.5% |
| 100k steps | **1.16** | ~85% |

Only ~7% better from 5× the compute: the ~82–86% recovery level is a property of the current
objective (the VPD baseline plateaus in the same range), not undertraining.

**Do the components mean anything?** (open-ended inspection of the 100k best checkpoint; ~2,800 of
4,096 components fire somewhere; verified causally by deleting one component and seeing which
predictions degrade)

| firing rate band | what they are | examples |
|---|---|---|
| always-on (~6 comps) | generic text machinery, polysemantic, heavy causal load | punctuation/markup engine |
| common (2–20%) | individual syntax rules | "after a preposition, predict *the*" (largest single causal effect found) |
| rare (<2%) | format & topic specialists | the word "systems" (deleting it damages *systems/machines/technology* predictions); LaTeX math; academic citation markup; code close-brackets; biomedical vocabulary |
| residue | near-dead duplicates, no causal role | a few `the`/comma detectors |

Meaning sharpened substantially with training budget: at 20k steps the rare band was mostly
duplicate punctuation detectors; at 100k it contains the specialists above.

## Caveats that apply throughout

- VPD numbers are our reimplementation/training of the published method on our targets (the paper
  itself reports on a different, larger model); both methods are always measured with identical
  code on identical targets.
- Toy results: 3 seeds for the distributed-reps recovery row; other rows single-seed. Pythia main
  comparison: 2 seeds; frontier and 100k rows single-run.
- Component interpretations are human-written glosses of firing/ablation statistics; automated
  labeling (`auto_interp.py`) is built but not yet run at scale.
