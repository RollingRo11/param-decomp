**Fellow:** Rohan Kathuria
**Mentor:** Gabriele Sarti (BauLab, Northeastern University)

# Introduction
Parameter decomposition is a currently under-explored endeavor in mechanistic interpretability (with the exception of Goodfire's parameter decomposition team). It aims to decompose the parameters of a neural network, rather than attempting to interpret it's activations. Ideally, we end up with known subnetworks of the model that are responsible for a specific output.

If successful, parameter decomposition would incredible benefits for the fields of AI safety and control. Namely, parameters are easy to edit (see [this](https://www.lesswrong.com/posts/ieoWstubDQWLrMnhH/exploration-fine-tuning-with-parameter-decomposition) recent post by a member of Goodfire's parameter decomposition team, Lucius Bushnaq, on granularly removing an LM's ability to speak German without affecting other languages.)

# Overview
To review, we set out to develop a parameter decomposition method that
1) Is able to find larger, more complex mechanisms without a post-hoc clustering method
2) Where these parameter components are not restricted to a single layer or matrix, and can properly express mechanisms implemented across layers

The benefits achieving these two goals provides are:
- **Better accuracy:** as shown in the results section, it remains to be proven or shown that a post-hoc clustering method of small subcomponents can accurately interpret a neural network's cross-layer mechanisms correctly.
- **Better actionability:** whole components that encompass more of the internal "machinery" of the model required for a specific output are more actionable and easy to interpret than hundreds or thousands of shattered subcomponents.
- **More faithful:** models implement behaviors as circuits spanning several layers; a unit of decomposition confined to one matrix cannot even *express* such a mechanism as one object.

Since submitting my report, I've developed the following method aimed at model-depth decomposition of deep learning models.

# Parameter Decomposition Methodology

**APD** (Braun et al. 2025) proposed the right *shape*: a component is a thin slice of every weight matrix at once, so a cross-layer mechanism is one object — but its attribution-plus-top-k training was unstable. **SPD/VPD** (Bushnaq et al. 2025; Goodfire 2026) contributed the right *training machinery* — a small side network learns which components matter per input, verified by actually deleting components — but its unit is a rank-1 piece of a *single* matrix, so cross-layer mechanisms shatter and must be reassembled by post-hoc clustering. Our method combines the two, and adds one new ingredient: **variable rank**.

## The decomposition object

For weight matrices $W^{(1)}, \dots, W^{(M)}$ (every attention and MLP projection), we create $C$ **components**. Component $c$ owns a piece of *every* matrix, each a sum of at most $R_m$ rank-1 terms:

$$P_c^{(m)} = \sum_{r=1}^{R_m} a_{c,r}^{(m)} \big(b_{c,r}^{(m)}\big)^{\!\top}, \qquad \text{with } \; \mathcal{L}_{\text{faith}} = \sum_m \Big\| W^{(m)} - \sum_c P_c^{(m)} \Big\|_F^2$$

enforcing that components sum back to the network exactly (**faithfulness**). Each component has **one gate** $g_c(x) \in [0,1]$, computed per token by a small side network, shared across all the component's pieces — this sharing is what makes it a whole-network object. A masked forward runs the model with $W_g^{(m)}(x) = \sum_c g_c(x) P_c^{(m)}$, so $g_c = 0$ deletes component $c$ everywhere at once: the gate is a *promise the component can be deleted without changing the output*, and training checks that promise by performing deletions.

## Training losses

- **Stochastic reconstruction** — delete components at random in proportion to their unimportance and match the original output: $\mathcal{L}_{\text{stoch}} = D\big(f_{\,g+(1-g)\odot u}(x), f(x)\big)$, $u \sim U(0,1)$, $D$ = KL for LMs. Every component is sampled every step, so dormant ones keep receiving gradient.
- **Adversarial reconstruction** — an attacker searches (gradient steps) for the worst combination of "unimportant" components to delete; penalize that damage.
- **Importance minimality** — $\sum_c g_c(x)^p$, $p$ annealed $2 \to 0.4$: an increasingly literal count of active components.
- **Hidden reconstruction** — the deletion-tested model must also match the original's *intermediate* activations, not just the final output.

## Variable rank

With caps $R_m > 1$, training pressure decides how much rank each component uses:

- **Nested ranks.** Each step, draw a cutoff $k \in \{1,2,4,\dots,R\}$ and run the stochastic reconstruction with every component truncated to its first $k$ terms (faithfulness and evals use the full component). Components can't know how many terms they'll get, so terms become importance-ordered and unused rank dies in the tail. Packing two mechanisms into one component is punished automatically — the cutoff regularly truncates the second one while its gate is up.
- **Rank trim.** Term sizes $s_{c,r}^{(m)} = \|a\| \|b\|$ are penalized as $\sum (\rho_0 + \rho_c)\, (s^2 + \epsilon^2)^{p/2}$ with $p = 0.5$ and $\rho_c$ the component's firing rate: sub-linear (counts terms rather than shrinking them), and usage-weighted so frequent components stay small while rare specialists may be large.

The textbook alternative — a Frobenius/nuclear-norm penalty — fails: under faithfulness its optimum is a few fat merged components, the exact failure it should prevent. This is close to APD's original simplicity penalty and explains part of its instability.

# Results

Baseline throughout: **VPD** (rank-1 per-matrix pieces, independent gates), trained by us with identical code, targets, and budgets. Metrics: **CE-recovered** — keep only gate-selected components; share of performance surviving (100% = perfect). **KL (masked)** — same test as output-distribution divergence. **KL (all-on)** — nothing deleted; must be ≈0 or the sum isn't faithful. **Separation** — each known mechanism gets its own component; **coverage** — all mechanisms get one. **Keep-only error** — delete all *but* a mechanism's component; does it still work ("do nothing" ≈ 0.6).

## Toy Model of Superposition

5 features in 2 dimensions, one rank-1 mechanism each: mean cosine similarity to ground truth **0.982** vs VPD's 0.927 (APD reports 0.998), with ~0.9 components active per input (correct: ~1 active feature). The sanity floor — one matrix, so the whole-network gate is untested here.

## Cross Layer Mechanisms

A 2-layer residual MLP (from the APD/SPD line) with 100 known mechanisms, each deliberately spread across both layers. VPD needs a post-hoc clustering step (knob α):

| method | separation | coverage | cross-layer | keep-only ↓ |
|---|---|---|---|---|
| VPD + clustering, α=1 | 0.66 | — | 0.46 | 0.22 |
| VPD + clustering, α=10 | 0.90 | — | **0.06** | 0.37 |
| **ours, rank-1** (3 seeds) | **0.89–0.92** | **0.97–1.00** | **1.00** | **0.10–0.12** |

The baseline hits a wall: clustering buys separation only by destroying cross-layer structure — no α gives both. Ours gets both at once with no clustering, because the shared gate makes cross-layer grouping learned during training. A structural capability difference, not a tuning difference.

**Variable-rank ablations** (cap 8; ground truth rank 1, so the method should *choose* ≈1):

| variant | separation | coverage | true rank (median) |
|---|---|---|---|
| cap only, no rank pressure | 0.68 | 0.99 | at cap |
| Frobenius penalty | 0.17–0.61 | — | inert or merging |
| nested alone | 0.91 | 0.99 | ~3 |
| **nested + trim** | 0.90 | 0.93 | **2** |

Rank freedom alone packs mechanisms together; nesting un-packs them; the trim removes the tail. ("True rank" = SVD of the materialized component.)

## Scaling to a real language model

Pythia-14M, all 24 matrices decomposed, ~800M tokens per run on 2×H100:

| | VPD | ours, rank-1 (best) | **ours, nested+trim (cap 8, best)** |
|---|---|---|---|
| CE-recovered | 82.2% | ~85% | **91.8%** |
| KL, masked ↓ | 1.46 | 1.16 | **0.59** |
| KL, all-on ↓ | 0.014 | ~1e-5 | 0.0014 |
| adversarial KL ↓ | **3.6** | 35.4 | 23.0 |
| gates active/token | 0.72% | 0.54% | 1.5% |

At rank-1 we already match the baseline's sparsity and beat its reconstruction (2 seeds). The headline: both methods previously plateaued at ~82–86% recovery, budget-independently (5× more training bought ~7% relative). Variable rank **breaks the ceiling** — it was a rank-1 expressiveness limit, not an objective limit. Adversarial robustness improved ~35% but remains our deficit. Anatomy: only 4 always-on components (~25% of weight energy) — no mega-component formed; MLP pieces saturate their cap while attention differentiates, motivating the per-type caps now running.

**Do the components mean anything?** Ranking by causal damage-when-deleted (gate magnitude surfaces dead duplicates instead), the top census entries are individual grammar rules: an "an"→vowel-initial-word component, a preposition→"the" component (largest single causal effect), a subject/verb agreement family, a sentence-boundary component, a code-indentation component. Main flaw exposed: **redundancy** — the "an" rule exists in three near-identical copies; our anti-redundancy penalty works on a toy but not yet at LM scale.

# Challenges

## Cost

Every step runs the original, a randomly-deleted, and an adversarially-deleted model, and the component bank multiplies parameters — the Pythia runs are ~10–20 GPU-hours each on 2×H100. Mid-project we made the loop ~3.2× faster (tensor-core math, mixed-precision forwards with fp32 losses, bucketed gradient sync), which makes the remaining controls cheap. Scaling past ~100M parameters is an engineering problem, not a conceptual one, but real.

## The alternative we tested and rejected: attribution-routed training

Could attribution (integrated-gradients / Shapley-style credit) replace the trained gate, as in original APD? We built the strongest version we could — interaction-aware subset attribution, sparse-but-differentiable selection instead of top-k, a causal-role simplicity criterion, all supporting terms matched. It recovers cross-layer structure but cannot match the trained mask, and at matched training pressure it collapses — three failure modes, one root cause: **attribution must route on component structure, but components only develop structure if routing already concentrates gradient** — a bootstrap circle only a co-trained gate escapes. We believe this reproduces why original APD was unstable (not its top-k). The negative result answers "why this hybrid": APD's shape and VPD's training are each doing irreplaceable work. Attribution stays useful as an *analysis* tool on trained decompositions.

## The honest limit: dense, always-on mechanisms

On mechanisms every input exercises — an induction-only toy transformer, and the text-copying machinery of real LMs — **no method variant we tested forms a dedicated component or crew**: not ours, not our VPD baseline, not the published VPD decomposition of a 67M model we verified has a single strong induction head. There, the mechanism is recoverable only as a diffuse population of hundreds of pieces whose joint deletion degrades copying (verified against matched random controls), while deleting the twenty pieces owning most of the head's weight changes almost nothing. Sparsity and capacity pressures are blind here because these mechanisms differ from the backbone by *role*, not firing rate or size — designing a role-based pressure is the clearest next problem.

## Runs in flight

1. **C=4096 with per-type caps (attention 8, MLP 16)**, full token budget.
2. **C=512, cap 8 — the rank-1 flagship's exact piece budget**: the control separating "variable rank helps" from "more parameters help".
3. **A second seed of the 91.8% run** (currently single-seed).
