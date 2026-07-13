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
- **More faithful:** Our current knowledge of neural networks provided by interpretability shows us that models implement their behaviors as circuits spanning several layers — attention heads reading what earlier layers wrote, MLPs refining it. A unit of decomposition confined to one matrix cannot even *express* such a mechanism as one object, so the honest unit must span the model's depth.

Since submitting my report, I've developed the following method aimed at model-depth decomposition of deep learning models.

# Parameter Decomposition Methodology

The two prior methods bracket what we want. **APD** (Braun et al. 2025) proposed the right *shape*: a component is a thin slice of every weight matrix at once, so a cross-layer mechanism is captured as a single object. But its training (gradient-based attribution with hard top-k selection) was unstable — an instability we later reproduced and root-caused (see Challenges). **SPD/VPD** (Bushnaq et al. 2025; Goodfire 2026) contributed the right *training machinery*: a small side network learns, for every input, which components matter, and is held to its answers by actually deleting components and checking the model still works. But their unit is a rank-1 piece of a single matrix with its own switch, so cross-layer mechanisms shatter across many units and must be reassembled by post-hoc clustering.

Our method combines the two — APD's whole-network component shape, trained with the SPD/VPD deletion-based machinery — and adds one new structural ingredient developed during this project: **variable rank**.

## The decomposition object

Let the network have weight matrices $W^{(1)}, \dots, W^{(M)}$ (for a transformer: every attention and MLP projection). We create $C$ **components**. Component $c$ owns one piece of *every* matrix, and each piece is a sum of at most $R_m$ rank-1 terms (each an outer product of a read direction $b$ and a write direction $a$):

$$P_c^{(m)} \;=\; \sum_{r=1}^{R_m} a_{c,r}^{(m)}\, \big(b_{c,r}^{(m)}\big)^{\!\top}, \qquad c = 1,\dots,C,\quad m = 1,\dots,M.$$

Conceptually (following APD), think of all decomposed weights flattened into one long parameter vector: a component is one vector in that space, and the $C$ components must sum to the network's parameter vector exactly. This **faithfulness** constraint is enforced as a hard loss:

$$\mathcal{L}_{\text{faith}} \;=\; \sum_{m} \Big\| W^{(m)} - \sum_{c} P_c^{(m)} \Big\|_F^2 .$$

Each component has **one gate** $g_c(x) \in [0,1]$, computed per token by a small side network reading the activations that flow into the decomposed matrices. The same gate applies to all of the component's pieces across all matrices — that sharing is what makes a component a whole-network object, and it is the entire difference from the per-matrix baseline. A masked forward pass runs the model with every matrix replaced by

$$W_g^{(m)}(x) \;=\; \sum_{c} g_c(x)\, P_c^{(m)},$$

so setting $g_c = 0$ deletes component $c$ from the whole network at once. A gate of 0 is a *promise that the component can be deleted on this input without changing the output* — and the training losses check that promise by actually performing deletions.

## Training losses

Each loss, in one sentence of the form "penalizes ___ by doing ___":

- **Stochastic reconstruction** — penalizes the gate calling a component unimportant when the model needs it, by deleting components at random in proportion to their unimportance and comparing outputs to the original model. With $u_c \sim U(0,1)$:
$$\mathcal{L}_{\text{stoch}} = D\!\Big(f_{\,g + (1-g)\odot u}(x),\; f(x)\Big),$$
where $D$ is the KL divergence between output distributions for language models. Because every component is randomly sampled into the forward pass every step, dormant components keep receiving gradient and can be recruited — a property we later showed is load-bearing.
- **Adversarial reconstruction** — penalizes weaknesses that random deletion would miss, by letting an attacker search (projected gradient steps) for the most damaging combination of "unimportant" components to delete, and penalizing that worst-case damage.
- **Importance minimality** — penalizes using many components at once: $\sum_c g_c(x)^p$ with the exponent $p$ annealed from 2 toward 0.4, so it starts gentle and becomes an increasingly literal count of active components.
- **Hidden reconstruction** — penalizes producing the right output the wrong way, by requiring the deletion-tested model to also match the original's *intermediate* activations at each decomposed matrix's output, not just the final answer.

## Variable rank: nested ordering plus a usage-coupled trim

With rank caps $R_m > 1$ (rank is only defined per matrix, and we allow different caps for attention vs MLP matrices), something must decide how much rank each component actually uses — and must prevent one component from hoarding capacity. Two pressures, both validated by ablation below:

- **Nested ranks.** Each step we draw a cutoff $k \in \{1, 2, 4, \dots, R\}$ and run the stochastic reconstruction with every component truncated to its **first $k$ rank-1 terms** (faithfulness and all evaluations use the full component). Since a component never knows how many terms it will be allowed, the only low-loss arrangement is importance-ordered: term 1 carries the mechanism's core, later terms refine, unused rank dies in the tail. Nesting also directly punishes packing two mechanisms into one component: whenever the cutoff truncates the second mechanism, its gate is up but the mechanism is missing, and the loss is paid — two dedicated components pay nothing.
- **Rank trim.** Each rank-1 term has a gauge-invariant size $s_{c,r}^{(m)} = \|a_{c,r}^{(m)}\| \cdot \|b_{c,r}^{(m)}\|$; we penalize
$$\mathcal{L}_{\text{trim}} = \sum_{m,c} (\rho_0 + \rho_c) \sum_r \big(s_{c,r}^{(m)\,2} + \epsilon^2\big)^{p/2}, \qquad p = 0.5,$$
with $\rho_c$ the component's firing rate and $\rho_0$ a small floor. Sub-linear in $s$, so it counts terms rather than shrinking them all; the usage weight makes frequent components stay low-rank while rare specialists may be large — big *and* always-on is the one configuration taxed by both factors.

A natural alternative — penalizing the factors' Frobenius norm, the textbook convex surrogate for low rank — **fails**: under the faithfulness constraint its optimum is a few fat merged components, i.e. it causes the exact failure it was meant to prevent. We report this because it is the closest thing to APD's original "simplicity" penalty, and it explains part of APD's instability.

# Results

All baselines are **VPD** (rank-1 per-matrix pieces, one independent gate each), trained by us with identical code on identical targets at matched budgets. Metrics in one line each: **CE-recovered** — keep only the components the gate marks important; what share of task performance survives (100% = perfect). **KL (masked)** — the same test as a divergence between output distributions (0 = perfect). **KL (all-on)** — sanity check with nothing deleted; must be ≈0 or the components don't even sum to the model. **Separation** — on toys with known mechanisms, does each mechanism get its own dedicated component; **coverage** — do all mechanisms get one (always read together). **Keep-only error** — delete everything *except* a mechanism's component; does the mechanism still work (lower is better; "do nothing" ≈ 0.6).

## Toy Model of Superposition

On the classical toy model of superposition (5 features stored in 2 dimensions; ground truth is one rank-1 mechanism per feature), our method recovers the known mechanisms nearly perfectly: mean cosine similarity to ground truth **0.982**, vs **0.927** for the matched VPD baseline (APD's own machinery reports 0.998 on this task). Reconstruction error 0.003 with ~0.9 components active per input — correct, since ~1 feature is active per input. This is the sanity floor: a single decomposed matrix, so it does not yet test the whole-network gate.

## Cross Layer Mechanisms

To show that this method accurately captures cross-layer mechanisms more effectively, we utilize the 2-layer residual MLP from the APD/SPD line of work: 100 known mechanisms, each deliberately spread across both layers. A perfect decomposition has one component per mechanism, spanning both layers. VPD's per-matrix pieces cannot express this directly, so the published recipe adds a clustering step afterwards (knob α = how finely to split).

| method | separation | coverage | cross-layer | keep-only ↓ |
|---|---|---|---|---|
| VPD + clustering, α=1 | 0.66 | — | 0.46 | 0.22 |
| VPD + clustering, α=10 | 0.90 | — | **0.06** | 0.37 |
| **ours, rank-1** (3 seeds) | **0.89–0.92** | **0.97–1.00** | **1.00** | **0.10–0.12** |

The baseline hits a wall: its clustering knob buys separation only by destroying cross-layer structure — no setting of α gives both. Our method gets both simultaneously, with no clustering step, because the shared gate makes cross-layer grouping something *learned during training*. This is a structural capability difference, not a tuning difference, and it is the strongest single piece of evidence for goal (2).

**Variable-rank ablations on the same toy** (rank cap 8; ground truth is rank 1, so the method should *choose* rank ≈ 1):

| variant | separation | coverage | true rank (median) |
|---|---|---|---|
| cap only, no rank pressure | 0.68 | 0.99 | at cap |
| Frobenius penalty | 0.17–0.61 | — | inert or merging |
| trim alone | 0.64 | 0.96 | ~3 |
| nested alone (4× budget) | 0.91 | 0.99 | ~3 |
| **nested + trim** | 0.90 | 0.93 | **2** (80/130 components at ≤2) |

Rank freedom with no counter-pressure quietly packs several mechanisms into one component's budget (separation drops to 0.68); nested ordering un-packs them; the trim removes the leftover tail. ("True rank" is measured by SVD of the materialized component — counting rank-1 terms overestimates, since terms can mix and cancel.)

## Scaling to a real language model

Target: **Pythia-14M** (6 layers, MLPs included, Pile-trained), all 24 weight matrices decomposed, ~800M training tokens per run on 2×H100.

**Matched-budget comparison, and the effect of variable rank:**

| | VPD baseline | ours, rank-1 (C=4096, best) | **ours, nested+trim (C=1024, cap 8, best)** |
|---|---|---|---|
| CE-recovered | 82.2% | ~85% | **91.8%** |
| KL, masked ↓ | 1.46 | 1.16 | **0.59** |
| KL, all-on (sanity) ↓ | 0.014 | ~1e-5 | 0.0014 |
| adversarial KL ↓ | **3.6** | 35.4 | 23.0 |
| gates active per token | 0.72% | 0.54% | 1.5% |

Two findings. First, at rank-1 our method already matches the baseline's sparsity while beating its reconstruction (replicated across 2 seeds). Second — the headline — both methods previously plateaued at ~82–86% recovery, and we had established that plateau was budget-independent (5× more training bought ~7% relative). Variable rank **breaks the ceiling**: 91.8% CE-recovered with masked KL halved. The ceiling was a rank-1 expressiveness limit, not a limit of the training objective. Worst-case (adversarial) robustness also improved ~35%, though it remains our one deficit against the baseline.

Rank anatomy of the trained model: only 4 always-on components holding ~25% of weight energy — the "mega-component" failure the caps and trim exist to prevent did not occur. MLP pieces saturate their cap of 8 in every usage band while attention components differentiate below it (rare ones use ~4) — evidence that MLP mechanisms want more rank, which motivated per-matrix-type caps (attention 8, MLP 16) in the run currently in flight.

**Do the components mean anything?** Ranking components by how much deleting each one damages the model *at the positions where it fires* (a causal criterion — picking by gate magnitude surfaces dead duplicates instead), the top of the census contains individual grammar rules: an indefinite-article component (fires on "an", causally supports vowel-initial continuations — *important, early, example*); a preposition→"the" component (the largest single causal effect found); a subject/verb agreement family (*is/are/was/be* after the corresponding subjects and auxiliaries); a sentence-boundary component (fires on `.`/`?`, supports newline plus capitalized sentence-openers); a code-indentation component. The census also exposes the main structural flaw: **redundancy** — the "an" rule exists as three near-identical copies. Our anti-redundancy penalty works on a toy but is not yet effective at LM scale; it is a known open problem.

# Challenges

## Cost

Decomposition training is several times the cost of ordinary training on the same tokens: every step runs the original model, a randomly-deleted copy, and an adversarially-deleted copy, and the component bank multiplies parameters by roughly $C \times R / \text{rank}(W)$. Concretely, the Pythia-14M runs above are ~10–20 GPU-hours each on 2×H100 for ~800M tokens. Mid-project we made the training loop ~3.2× faster (tensor-core math, mixed-precision forwards with all losses kept in fp32, single-bucket gradient synchronization), validated against fp32 trajectories before adoption; this is what makes the remaining control runs cheap. Scaling beyond ~100M-parameter targets is an engineering problem (sharded component banks), not a conceptual one, but it is real.

## The alternative we tested and rejected: attribution-routed training

Since our method inherits VPD's *trained* gate network, we asked whether the gate could instead be computed directly by **attribution** (integrated-gradients / Shapley-style credit for each component), which would be closer to original APD and remove a learned module. We built the strongest version we could: interaction-aware random-subset attribution, a sparse-but-differentiable selection (replacing APD's hard top-k), a causal-role simplicity criterion (replacing its rank penalty), and every supporting term matched to the trained-mask stack. It recovers cross-layer structure and decent separation (up to 0.79) — but cannot match the trained mask, and at matched training pressure it collapses, in three distinct ways that share one root cause: **attribution must route on the components' structure, but components only develop structure if routing already concentrates gradient on them** — a bootstrap circle that only a co-trained, expressive gate escapes. We believe this reproduces, from first principles, why original APD was unstable (it was never the top-k). The negative result answers "why this hybrid": APD's component shape and VPD's training machinery are each doing irreplaceable work. Attribution remains valuable as an *analysis* tool on trained decompositions, where it cleanly separates genuine mechanism-carriers from high-weight but functionally inert pieces.

## The honest limit: dense, always-on mechanisms

On targets where every input exercises most of the network — a 2-layer induction-only transformer, and the text-copying (induction) machinery inside real LMs — **no method variant we tested forms a dedicated component or small crew for the mechanism**: not ours at any rank setting, not our VPD baseline, and not the published VPD decomposition of a 67M-parameter model that we verified has a single strong induction head. In that 67M model the mechanism is recoverable only as a diffuse population of several hundred small pieces whose *joint* deletion degrades copying (verified causally against size- and rate-matched random controls), while deleting the twenty pieces that own most of the induction head's weight changes almost nothing. Our diagnosis: sparsity- and capacity-based pressures are structurally blind here, because these mechanisms differ from the always-on backbone by *role*, not by firing rate or size. Designing a role-based training pressure is the clearest next problem this project has surfaced.

## Runs in flight

1. **C=4096 with per-type rank caps (attention 8, MLP 16) at the full 800M-token budget** — more component slots plus the wider MLP cap the anatomy called for.
2. **C=512, cap 8 — exactly the rank-1 flagship's total piece budget** — the control separating "variable rank helps" from "more parameters help"; if it still beats the rank-1 flagship, the ceiling break is attributable to rank structure specifically.
3. **A second seed of the 91.8% run** (currently single-seed; the rank-1 rows are 2–3 seeds).
