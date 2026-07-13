**Fellow:** Rohan Kathuria
**Mentor:** Gabriele Sarti (BauLab, Northeastern University)

# Introduction
Parameter decomposition is a currently under-explored endeavor in mechanistic interpretability (with the exception of Goodfire's parameter decomposition team). It aims to decompose the parameters of a neural network, rather than attempting to interpret it's activations. Ideally, we end up with known subnetworks of the model that are responsible for a specific output.

If successful, parameter decomposition would yield incredible benefits for the fields of AI safety and control. Namely, parameters are easy to edit (see [this](https://www.lesswrong.com/posts/ieoWstubDQWLrMnhH/exploration-fine-tuning-with-parameter-decomposition) recent post by a member of Goodfire's parameter decomposition team, Lucius Bushnaq, on granularly removing an LM's ability to speak German without affecting other languages.)

# Overview
To review, we set out to develop a parameter decomposition method that
1) Is able to find larger, more complex mechanisms without a post-hoc clustering method
2) Where these parameter components are not restricted to a single layer or matrix, and can properly express mechanisms implemented across layers

# Method

The original landmark parameter decomposition method, Attribution Based Parameter Decomposition (**APD**) (Braun et al. 2025) had the right *shape*: a component spans every weight matrix, so a cross-layer mechanism is one object, however the authors found training under an attribution based setup too unstable.

Newer work from the same authors, stochastic and adVersarial parameter decomposition methods **SPD/VPD** (Bushnaq et al. 2025; Goodfire 2026) have the right *training*: a small network learns which components matter per input, verified by actually deleting them. They also decompose the model into rank-1 *subcomponents*, limited to individual matrices that they then hope to later cluster into whole components.

The above motivates our method, which combines the two and adds one new ingredient: **variable rank**.

For weight matrices $W^{(1)}, \dots, W^{(M)}$ we create $C$ components, each owning a piece of *every* matrix, built from at most $R$ rank-1 terms:

$$P_c^{(m)} = \sum_{r=1}^{R} a_{c,r}^{(m)} \big(b_{c,r}^{(m)}\big)^{\!\top}, \qquad \mathcal{L}_{\text{faith}} = \sum_m \Big\| W^{(m)} - \sum_c P_c^{(m)} \Big\|_F^2 .$$

Each component has **one gate** $g_c(x) \in [0,1]$ (a small side network, per token) shared by all its pieces — that sharing is what makes it a whole-network object. The model runs with $W_g^{(m)}(x) = \sum_c g_c(x) P_c^{(m)}$, so $g_c=0$ deletes the component everywhere at once. A gate of 0 is a promise the component can be deleted without changing the output, and each loss checks that promise a different way:

- **Faithfulness** ($\mathcal{L}_{\text{faith}}$ above) — the components must sum back to the original weights exactly, or nothing else is meaningful.
- **Stochastic reconstruction** — delete components at random, in proportion to how unimportant the gate claims they are, and require the output to match the original model:
$$\mathcal{L}_{\text{stoch}} = D_{\mathrm{KL}}\!\Big(f_{\,g + (1-g)\odot u}(x)\;\big\|\; f(x)\Big), \qquad u \sim U(0,1)^C .$$
- **Adversarial reconstruction** — an attacker searches (by gradient steps) for the *worst* deletion of "unimportant" components; penalize the damage it finds:
$$\mathcal{L}_{\text{adv}} = \max_{s \,\in\, [0,1]^C}\; D_{\mathrm{KL}}\!\Big(f_{\,g + (1-g)\odot s}(x)\;\big\|\; f(x)\Big).$$
- **Hidden reconstruction** — the deletion-tested model must also match the original's *intermediate* activations $h^{(m)}$ at every decomposed matrix, not just the final output:
$$\mathcal{L}_{\text{hidden}} = \tfrac{1}{M}\sum_m \big\| h_g^{(m)}(x) - h^{(m)}(x) \big\|^2 \,/\, \mathrm{Var}\big(h^{(m)}\big).$$
- **Importance minimality** — few components should be active per token; the exponent anneals from 2 toward 0.4, becoming an increasingly literal count of active components:
$$\mathcal{L}_{\text{min}} = \sum_c g_c(x)^p, \qquad p: 2 \to 0.4 .$$

**Variable rank:** each step, all components are truncated to their first $k$ rank-1 terms (with a random $k$) during reconstruction — so terms become importance-ordered and unused rank dies — plus a small usage-weighted penalty on term sizes $s_{c,r}^{(m)} = \|a_{c,r}^{(m)}\|\|b_{c,r}^{(m)}\|$,
$$\mathcal{L}_{\text{trim}} = \sum_{m,c} (\rho_0 + \rho_c) \sum_r \big(s_{c,r}^{(m)\,2} + \epsilon^2\big)^{p/2}, \qquad p = 0.5,$$
with $\rho_c$ the component's firing rate: frequent components stay small, and only rare specialists may be large.

# Progress & Results

All comparisons are against VPD, trained by us with identical code, targets, and budgets. **CE-recovered** asks the question, "when you keep only the selected components; what share of performance survives?"

**Cross-layer mechanisms.** On a 2-layer residual MLP with 100 known mechanisms deliberately spread across both layers (see Bushnaq et al. 2025), VPD's own clustering hits a wall: it reaches 0.90 separation (one mechanism per cluster) only by collapsing cross-layer structure to 0.06, or keeps cross-layer structure at the cost of separation.  

Our method gets both at once (separation 0.89–0.92, cross-layer 1.0), and deleting everything *except* one mechanism's component leaves that mechanism working. Structurally, this is impossible for VPD, and even with their clustering step it cannot properly combine these known mechanisms into labeled ones.

On a real language model, Pythia-14M, if we make our cross-layer components rank-1, we match VPD's sparsity and reconstruction. However, when we make our components variable rank, we beat reconstruction by ~10% (82.2 vs 91.8), and cut the KL-divergence in half, yielding a better on-paper decomposition while keeping interpretability. We also use half the amount of components on average to explain some output under the same language model.

**Interpretability.**  When we rank components by how causally important they are to ablate, we find a variety of interpretable components with grammar rules, such as "an"→vowel-initial words; preposition→"the" (largest single effect); subject/verb agreement; sentence boundaries; code indentation.

It remains to be seen how much more interpretable our cross-layer, variable rank version of the training setup is compared to VPD on equal footing; that is, their published result on a 67M parameter pile language model. Our next large objective is to scale our method to this model, and compare their subcomponents+clustering to our components.

# What I've learned

- **You can control how big a component is, but no size penalty decides what job it does.** Whenever we gave components more room without other pressure, they quietly took on several unrelated jobs at once — and every penalty on size or rank either did nothing or made components merge. What kept one job per component was the ordering trick in the variable-rank setup, not any size knob. Lesson: "how big" and "what for" are separate problems, and most of the field's tools only address the first.
- **Why the original APD was unstable.** We rebuilt APD's attribution-based training with every modern fix we could think of, and it still fell apart — for a reason that turned out to be circular. To decide which components matter for an input, you need components that already do distinct jobs; but components only *learn* distinct jobs if that decision is already being made well. A gate network that trains alongside the components escapes the circle, because both sides improve together; a fixed importance formula cannot. This is also the answer to "why this particular hybrid": APD's component shape and VPD's trained gate each do work the other cannot.
- **The honest limit: dense mechanisms.** For machinery every input uses (e.g. induction/copying), no variant we tested — ours, our VPD baseline, or the paper's own decomposition of a 67M model with one strong induction head — forms a dedicated component: the mechanism stays smeared across hundreds of pieces (verified causally). These mechanisms differ from the backbone by *role*, not firing rate or size, and no current loss sees role. That's the clearest next problem.

# Challenges

Cost: each step runs the original, a randomly-deleted, and an adversarially-deleted model — ~10–20 GPU-hours per Pythia run. We made the loop ~3.2× faster mid-project (tensor cores, mixed precision, bucketed gradient sync). Scaling past ~100M parameters is engineering, not concept, but real.

# Final Stretch

In flight: a larger run with per-type rank caps (MLP mechanisms want more rank than attention); a parameter-matched control separating "variable rank helps" from "more parameters help"; a second seed of the 91.8% result. Then: make the anti-redundancy penalty work at LM scale, and automate the component census.

By the end of the fellowship I will have: the documented method (whole-network components + deletion-based training + variable rank), head-to-head results against VPD on toys and a real LM with the ceiling break seed-replicated and capacity-controlled, and a written account of the two negative results (attribution routing, dense mechanisms) mapping the method's boundaries. Final deliverable: the writeup with public code.
