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

- **Faithfulness** ($\mathcal{L}_{\text{faith}}$ above): the components must sum back to the original weights exactly, or nothing else is meaningful.
- **Stochastic reconstruction**: What happens if we delete components at random, in proportion to how unimportant the gate claims they are, and require the output to match the original model:
$$\mathcal{L}_{\text{stoch}} = D_{\mathrm{KL}}\!\Big(f_{\,g + (1-g)\odot u}(x)\;\big\|\; f(x)\Big), \qquad u \sim U(0,1)^C .$$
- **Adversarial reconstruction** (from VPD) an attacker searches (by gradient steps) for the *worst* deletion of "unimportant" components; penalize the damage it finds:
$$\mathcal{L}_{\text{adv}} = \max_{s \,\in\, [0,1]^C}\; D_{\mathrm{KL}}\!\Big(f_{\,g + (1-g)\odot s}(x)\;\big\|\; f(x)\Big).$$
- **Hidden reconstruction**: the deletion-tested model must also match the original's *intermediate* activations $h^{(m)}$ at every decomposed matrix, not just the final output. This is to prevent a wildly unfaithful decomposition:
$$\mathcal{L}_{\text{hidden}} = \tfrac{1}{M}\sum_m \big\| h_g^{(m)}(x) - h^{(m)}(x) \big\|^2 \,/\, \mathrm{Var}\big(h^{(m)}\big).$$
- **Importance minimality**: few components should be active per token; the exponent anneals from 2 toward 0.4, becoming an increasingly literal count of active components:
$$\mathcal{L}_{\text{min}} = \sum_c g_c(x)^p, \qquad p: 2 \to 0.4 .$$
**Variable rank:** each step, all components are truncated to their first $k$ rank-1 terms (with a random $k$) during reconstruction. Terms become importance-ordered and unused rank dies.
$$\mathcal{L}_{\text{trim}} = \sum_{m,c} (\rho_0 + \rho_c) \sum_r \big(s_{c,r}^{(m)\,2} + \epsilon^2\big)^{p/2}, \qquad p = 0.5,$$
with $\rho_c$ the component's firing rate: frequent components stay small, and only rare specialists may be large.

# Progress & Results

All comparisons are against VPD, trained by us with identical code, targets, and budgets. **CE-recovered** asks the question, "when you keep only the selected components; what share of performance survives?"

**Cross-layer mechanisms.** On a 2-layer residual MLP with 100 known mechanisms deliberately spread across both layers (see Bushnaq et al. 2025), VPD's own clustering hits a wall: it reaches 0.90 separation (one mechanism per cluster) only by collapsing cross-layer structure to 0.06, or keeps cross-layer structure at the cost of separation.  

Our method gets both at once (separation 0.89–0.92, cross-layer 1.0), and deleting everything *except* one mechanism's component leaves that mechanism working. Structurally, this is impossible for VPD, and even with their clustering step it cannot properly combine these known mechanisms into labeled ones.

On a real language model, Pythia-14M, if we make our cross-layer components rank-1, we match VPD's sparsity and reconstruction. However, when we make our components variable rank, we beat reconstruction by ~10% (82.2 vs 91.8), and cut the KL-divergence in half, yielding a better on-paper decomposition while keeping interpretability. We also use half the amount of components on average to explain some output under the same language model. Importantly, 96.6% of our components hold >10% of their weight in at least two layers, and there are zero single-layer components. 

**Interpretability.**  When we rank components by how causally important they are to ablate, we find a variety of interpretable components with grammar rules, such as "an"→vowel-initial words; preposition→"the" (largest single effect); subject/verb agreement; sentence boundaries; code indentation.

It remains to be seen how much more interpretable our cross-layer, variable rank version of the training setup is compared to VPD on equal footing; that is, their published result on a 67M parameter pile language model. Our next large objective is to scale our method to this model, and compare their subcomponents+clustering to our components.

# What I've learned
Grammatical mechanisms are dense for both our method and VPD. Circuits we'd hope to find cleanly, such as copying/induction, are currently unseen as one single component. Instead, there seems to be a wide set of grammatical components that aren't cleanly interpretable, though we find many that are. This may be because these grammatical mechanisms just aren't cleanly separable from others in small models. We'd have to scale our method to much larger models to show this.

# Challenges
- **Cost.** Decomposition training runs the original model, a randomly-deleted copy, and an adversarially-deleted copy every step, and because every component owns a piece of *every* matrix, our decomposition holds roughly 15–25× more parameters than VPD's per-matrix subcomponents on the same model. In measured wall-clock this comes out gentler than the parameter ratio suggests — roughly **2× VPD per training step at rank-1 and ~4× with variable rank** at matched token budgets (a Pythia-14M run is ~10–20 GPU-hours on 2×H100). The binding constraint is actually memory rather than compute: the per-token component activations scale with $C \times R$, which is what limits batch size as $C$ grows. We made the loop ~3.2× faster mid-project (tensor cores, mixed precision, bucketed gradient sync), but scaling past ~100M-parameter targets will need sharded component banks — engineering, not concept, but real.
- **Granularity.** There is a genuine tension between making components big enough to encompass whole circuits and granular enough to stay interpretable, and the objective has no term that knows where a "mechanism" ends. We see both failure directions in practice: too few components and the gates collapse to always-on generalists; too much capacity per component and unrelated jobs get packed together. Our own data shows the tension directly — the most *spread-out* components (weight in all 6 layers) are the polysemantic backbone, while the cleanly interpretable specialists are the most concentrated. Variable rank gives us a dial where before there was none, but what value of the dial equals "one mechanism" is still decided by us, not learned.
# Final Stretch
Over the next couple weeks, we aim to:
- Scale our method to the same, 67M parameter LM the VPD authors published their decomposition on.
- Meet with the authors of A/S/VPD and get their insights on attempting to scale this across layers and to larger models.
- Continue to iterate on making mechanisms more sparse and interpretable. Ideally, we're able to recover a full, known circuit (such as induction) with our method.
- Aim towards scaling to an even larger model (such as GPT2).


By the end of the summer term of the fellowship, we will have: the documented method, results against VPD with both toys, Pythia-14M, a 67M param LM, and a codebase with a tool to replicate this on other language models. Our final deliverable is a writeup (hopefully both in the form of a conference paper as well as an online interactive blogpost so viewers can explore components).

By the end of the fellowship I will have: the documented method (whole-network components + deletion-based training + variable rank), head-to-head results against VPD on toys and a real LM with the ceiling break seed-replicated and capacity-controlled, and a written account of the two negative results (attribution routing, dense mechanisms) mapping the method's boundaries. Final deliverable: the writeup with public code.
