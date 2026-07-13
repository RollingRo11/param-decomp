**Fellow:** Rohan Kathuria
**Mentor:** Gabriele Sarti (BauLab, Northeastern University)

# Introduction
Parameter decomposition is a currently under-explored endeavor in mechanistic interpretability (with the exception of Goodfire's parameter decomposition team). It aims to decompose the parameters of a neural network, rather than attempting to interpret it's activations. Ideally, we end up with known subnetworks of the model that are responsible for a specific output.

If successful, parameter decomposition would incredible benefits for the fields of AI safety and control. Namely, parameters are easy to edit (see [this](https://www.lesswrong.com/posts/ieoWstubDQWLrMnhH/exploration-fine-tuning-with-parameter-decomposition) recent post by a member of Goodfire's parameter decomposition team, Lucius Bushnaq, on granularly removing an LM's ability to speak German without affecting other languages.)

# Overview
To review, we set out to develop a parameter decomposition method that
1) Is able to find larger, more complex mechanisms without a post-hoc clustering method
2) Where these parameter components are not restricted to a single layer or matrix, and can properly express mechanisms implemented across layers

# Method

**APD** (Braun et al. 2025) had the right *shape*: a component spans every weight matrix, so a cross-layer mechanism is one object — but its training was unstable. **SPD/VPD** (Bushnaq et al. 2025; Goodfire 2026) has the right *training*: a small network learns which components matter per input, verified by actually deleting them — but its unit is a rank-1 piece of a single matrix, so mechanisms shatter and must be re-clustered post-hoc. My method combines the two, plus one new ingredient: **variable rank**.

For weight matrices $W^{(1)}, \dots, W^{(M)}$ we create $C$ components, each owning a piece of *every* matrix, built from at most $R$ rank-1 terms:

$$P_c^{(m)} = \sum_{r=1}^{R} a_{c,r}^{(m)} \big(b_{c,r}^{(m)}\big)^{\!\top}, \qquad \mathcal{L}_{\text{faith}} = \sum_m \Big\| W^{(m)} - \sum_c P_c^{(m)} \Big\|_F^2 .$$

Each component has **one gate** $g_c(x) \in [0,1]$ (a small side network, per token) shared by all its pieces — that sharing is what makes it a whole-network object. The model runs with $W_g^{(m)}(x) = \sum_c g_c(x) P_c^{(m)}$, so $g_c=0$ deletes the component everywhere at once. Training holds gates to that promise by performing deletions: randomly (match the output under random deletion of "unimportant" components), adversarially (an attacker picks the worst deletion), and at hidden layers (intermediate activations must match too), while a minimality term $\sum_c g_c^p$ ($p \to 0.4$) keeps few components active per token.

**Variable rank:** each step, all components are truncated to their first $k$ rank-1 terms ($k$ random) during reconstruction — so terms become importance-ordered and unused rank dies — plus a small usage-weighted penalty on term sizes, so frequent components stay small and only rare specialists may be large.

# Progress & Results

All comparisons are against VPD, trained by us with identical code, targets, and budgets. **CE-recovered** = keep only the gate-selected components; what share of performance survives.

**Cross-layer mechanisms.** On a 2-layer residual MLP with 100 known mechanisms deliberately spread across both layers, VPD's own clustering hits a wall: it reaches 0.90 separation (one mechanism per cluster) only by collapsing cross-layer structure to 0.06, or keeps cross-layer structure at the cost of separation — no knob setting gives both. Our method gets both at once (separation 0.89–0.92, cross-layer 1.0, 3 seeds, no clustering), and deleting everything *except* one mechanism's component leaves that mechanism working. This is a structural capability difference, and the strongest evidence for the whole-network design. (On the classical toy model of superposition we recover ground truth at 0.982 cosine similarity vs the baseline's 0.927.)

**Variable rank.** Same toy, rank cap 8, ground truth rank 1: rank freedom alone quietly packs several mechanisms into one component (separation falls to 0.68); nested truncation un-packs them (0.91) and the trim drives measured ranks to a median of 2. The textbook alternative (nuclear-norm penalty — essentially APD's original simplicity term) fails outright: its optimum under faithfulness *is* a few fat merged components.

**A real LM.** On Pythia-14M (all 24 matrices, ~800M tokens, 2×H100): at rank-1 we match VPD's sparsity and beat its reconstruction (82.6% vs 82.2% CE-recovered, 2 seeds). Both methods plateau at ~82–86% — and we showed the plateau is budget-independent (5× training bought ~7%). Variable rank **breaks the ceiling: 91.8% CE-recovered, masked KL halved**, same budget. The ceiling was a rank-1 expressiveness limit, not an objective limit. No mega-component formed (4 always-on components, ~25% of weight energy); worst-case robustness improved ~35% but remains our deficit vs VPD.

**Interpretability.** Ranking components by causal damage-when-deleted, the top of the census is individual grammar rules: "an"→vowel-initial words; preposition→"the" (largest single effect); subject/verb agreement; sentence boundaries; code indentation. Main exposed flaw: redundancy (the "an" rule exists in triplicate).

# What I've learned

- **Rank pressure controls how big components are, never what they are** — identity must come from structure (nesting) or role-based losses; every pure capacity penalty either did nothing or merged components.
- **Why original APD was unstable.** We rebuilt attribution-routed training in its strongest form (Shapley-style attribution, differentiable sparse selection, causal-role simplicity) and it still collapses at matched training pressure. Root cause: attribution must route on component structure, but structure only forms if routing already concentrates gradient — a bootstrap circle only a co-trained gate escapes. This answers "why this hybrid": APD's shape and VPD's training each do irreplaceable work.
- **The honest limit: dense mechanisms.** For machinery every input uses (e.g. induction/copying), no variant we tested — ours, our VPD baseline, or the paper's own decomposition of a 67M model with one strong induction head — forms a dedicated component: the mechanism stays smeared across hundreds of pieces (verified causally). These mechanisms differ from the backbone by *role*, not firing rate or size, and no current loss sees role. That's the clearest next problem.

# Challenges

Cost: each step runs the original, a randomly-deleted, and an adversarially-deleted model — ~10–20 GPU-hours per Pythia run. We made the loop ~3.2× faster mid-project (tensor cores, mixed precision, bucketed gradient sync). Scaling past ~100M parameters is engineering, not concept, but real.

# Final Stretch

In flight: a larger run with per-type rank caps (MLP mechanisms want more rank than attention); a parameter-matched control separating "variable rank helps" from "more parameters help"; a second seed of the 91.8% result. Then: make the anti-redundancy penalty work at LM scale, and automate the component census.

By the end of the fellowship I will have: the documented method (whole-network components + deletion-based training + variable rank), head-to-head results against VPD on toys and a real LM with the ceiling break seed-replicated and capacity-controlled, and a written account of the two negative results (attribution routing, dense mechanisms) mapping the method's boundaries. Final deliverable: the writeup with public code.
