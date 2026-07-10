# Whole-network parameter decomposition with a trained mask

*(files: `apd_mask.py` — toys + shared machinery; `apd_lm.py` — LM training loop (attn-only-2l
entry); `apd_pythia.py` — Pythia-14M entry; `apd_alg.py` — AlgZoo RNNs; `alg_interaction.py` /
`interact_pythia.py` / `interact_apd.py` — diagnostics and component interpretation. This doc
supersedes the Matryoshka writeup, now in `matryoshka_method.md`.)*

## 1. What we are trying to do

A trained neural network's behavior is produced by its **weights** — big matrices of numbers. We
want to split those weights into a set of **components**: separate pieces that (a) add back up to
the original weights exactly, and (b) each implement one reusable *mechanism* of the network, so
that a person can study, remove, or edit one mechanism at a time.

The method combines two ideas from prior work:

- From **Attribution-based Parameter Decomposition** (APD, Braun et al. 2025): the *shape* of the
  answer. A component is not a piece of one weight matrix — it is a thin slice of **every** weight
  matrix at once, so a mechanism that spans several layers is captured as a single object.
- From **Stochastic Parameter Decomposition** (SPD, Bushnaq et al. 2025) and its adversarial
  successor (VPD, Goodfire 2026): the *training machinery*. Instead of estimating which components
  matter with gradient heuristics (APD's approach, which proved brittle), we train a small side
  network that learns, for every input, which components matter — and we verify its answers by
  actually deleting components and checking the model still works.

The one-line contrast with SPD/VPD: their unit is a tiny piece of a *single* matrix with its own
on/off switch, so a cross-layer mechanism ends up scattered across many units and must be grouped
after the fact (which we showed fails on a toy with known cross-layer structure). Our unit spans the
whole network and has **one** switch, so cross-layer grouping is learned during training itself.

## 2. The decomposition object

For a chosen list of weight matrices (e.g. all attention projections of a transformer):

- We create **C components**. Conceptually (following APD), think of all the decomposed weights
  flattened into one long parameter vector: a component is one vector in that space, and the C
  components sum to the network's parameter vector exactly (enforced by a training loss). In
  implementation, component c is stored as one matrix-shaped piece per decomposed matrix — the same
  object, kept per-matrix because the structural constraints below only make sense per matrix.
  Nothing couples entries across matrices except the shared index and the shared gate; the gate is
  what makes the pieces one component (scaling component c scales all its pieces everywhere at
  once, and every loss treats it as one unit for that reason).
- Each piece is **rank-1 within its own matrix**: an outer product of two vectors (one direction
  reading in, one writing out). Note rank is only defined per matrix — there is no meaningful
  "rank of the concatenation of all matrices" — which is also why the simplicity of a cross-layer
  component is inherently harder to score than a single-layer one (APD flags this same
  layer-privileging issue). This is the strongest structural choice in the method, adopted after full-rank pieces
  repeatedly failed. A rank-1 piece is too small to secretly implement several mechanisms at once,
  which blocks the two failure modes that plagued richer components: one giant component that does
  everything, and many identical components that blur together. (The component count C must be at
  least the rank of the largest decomposed matrix, or the pieces cannot sum to it.)
- Each component has **one gate**: a number between 0 (this component is not used for this input)
  and 1 (it is needed). The same gate applies to all of the component's pieces across all matrices —
  that is what makes a component a whole-network object. Gates are computed *per position*: per
  token for a language model, per timestep for a recurrent network (where the same weight matrix is
  reused every step, so a "cross-layer" mechanism becomes a "cross-time" one).

## 3. The gate network

A small side network (the **causal-importance network**) reads the activations flowing *into* every
decomposed matrix and outputs the C gate values for each position. For contextual models it is a
small bidirectional transformer (whether a mechanism like "predict repeated text" is in use depends
on surrounding tokens, not just the current one); for simple toys a small MLP suffices. Its outputs
pass through a clipped-linear function so gates live in [0,1] with a slight slope outside, letting
gradients flow to gates that are currently pinned at 0 or 1.

"Matters" is defined causally, not by correlation: a gate value of 0 is a *promise that the
component can be deleted on this input without changing the model's output*. The training losses
below hold the gate network to that promise by actually performing the deletions.

## 4. The losses

Each loss is one sentence of the form "penalizes ___ by doing ___", with the details after.

**Faithfulness** — penalizes the components failing to add up to the original network, by measuring
the squared difference between the original weights and the sum of all components. This constraint
must be enforced *hard* (we use a coefficient of 1e7–1e8; when the L1 loss below is active the
higher value is needed, or the L1 quietly wins the tug-of-war and the components stop summing to the
model, invalidating everything else).

**Stochastic reconstruction** — penalizes the gate calling a component "unimportant" when the model
actually needs it, by randomly deleting components in proportion to how unimportant the gate says
they are, and measuring how much the output changes. (Where the gradient goes: the original network
is frozen and only supplies the target output. The loss trains the gate network — raising a gate
shrinks that component's exposure to deletion, so gates learn importance from the consequences of
deletions — and *simultaneously* reshapes the components, since computation can also be moved into
components the gates already protect. Combined with minimality pushing all gates down, this tension
is what forces each input's computation into a few protectable components — it is the mechanism by
which structure emerges.) The "measuring" must match the task: for
language models we compare output distributions (KL divergence); for regression-like toys we use
squared error scaled by the output's variance. We learned this the hard way: on a task decided by
tiny margins between top candidates (find the second-largest number), the distribution comparison
looked fine while the task answer was wrong constantly — the divergence must be sensitive to what
the task is actually sensitive to.

**Adversarial reconstruction** — penalizes weaknesses that random deletion would miss, by letting an
attacker search (by gradient steps) for the most damaging combination of "unimportant" components to
delete, and penalizing that worst-case damage. Without this, decompositions pass the random test
while hiding components that are secretly load-bearing. (Implementation: on the LM the attacker's
deletion pattern persists and updates across training steps; on toys it is re-derived fresh each
step. Both follow VPD.)

**Importance minimality** — penalizes using many components at once, by summing the gate values on
each input and pushing them toward zero, with an extra tax on components that fire on a large
fraction of inputs. The exponent on the gate values is annealed from 2 toward ~0.4–0.7 over training
(gentle at first, an increasingly literal "count of active components" later).

**Hidden-activation reconstruction** — penalizes components that produce the right output the wrong
way, by requiring the deletion-tested model to match the original's *intermediate* activations (each
decomposed matrix's output at every position, compared site-by-site inside the same masked forward
pass; for an RNN, the hidden state at every timestep), not just its final output. This shapes
internal routing and consistently improves worst-case robustness. For cross-layer components the
discipline is indirect but real: blame is localized at each layer's sites (downstream layers can no
longer compensate for upstream errors through the final output), and because a component's pieces
switch together under one gate, its pieces must be useful *as a pair* at their respective sites —
an accidental bundle whose layer-0 piece helps while its layer-1 piece hurts pays every time its
gate is up. (Within the masked pass, later sites receive masked earlier outputs — end-to-end
trajectory matching with compounding errors, following APD; the mask-one-layer-at-a-time
alternative, SPD's layerwise loss, tested neutral here as subset routing.)

**Interaction (anti-redundancy)** — penalizes pairs of components that act as backups for each
other. The problem it targets: nothing above stops two components from learning the *same*
mechanism. A redundant pair passes every other test — the sum is still faithful, and deleting
either one alone is harmless precisely *because* the other covers for it. Frequency-based penalties
can't see this either: the two backups fire at exactly the same rate as one honest component would.
The detection trick is to delete them **together**. If deleting A alone costs nothing, deleting B
alone costs nothing, but deleting A and B together is catastrophic, they were covering for each
other. Formally, each step we sample a few pairs and compute
*damage(both deleted) − damage(A deleted) − damage(B deleted) + damage(nothing deleted)*, and
penalize this quantity **only when it is positive** (deleting both is worse than the parts
predict = redundancy). When it is negative — deleting both is *less* bad than the parts predict —
the two components share a pipeline (A feeds B, so once A is gone, losing B adds little); that is
genuine structure, not redundancy, and is deliberately left alone. Mixed record: on the algorithmic
RNN this achieved causal independence at zero accuracy cost; on the LM it hurt at both doses tried
(§7).

**Entrywise sparsity (L1)** — penalizes carvings in which components overlap and cancel against
each other, thereby choosing *which of many equally-valid-looking carvings* we get. The problem it
targets: suppose the true mechanisms are M₁ and M₂, but the optimizer instead lands on
A = M₁ + M₂ and B = −M₂ (or any other mixture). The sum is unchanged, so faithfulness is satisfied;
with the gates co-adapted, the deletion tests can be satisfied too. Every loss above is blind to
the difference between the clean carving {M₁, M₂} and the mixed one {A, B} — that freedom is why
components can come out as plausible-looking mixtures ("blobs"). The fix exploits one arithmetic
fact: mixtures require **cancellation**. In the {A, B} carving, the M₂-entries appear twice with
opposite signs, so the total absolute value of all component entries is larger than in the clean
carving, where each weight entry appears in exactly one component. Since the components' *sum* is
pinned to the original weights, simply penalizing the total absolute value of all entries makes
clean, non-overlapping carvings the cheapest — it is the only loss in the method that breaks this
"which basis" tie. Free diagnostic: the ratio (total absolute component entries) / (total absolute
original entries) is exactly 1.0 for a perfectly non-overlapping carving, and larger the more the
components cancel against each other. The crucial caveat is that **some networks genuinely store
mechanisms as overlapping weight patterns** (superposition) — there the *correct* carving has ratio
well above 1 (measured ~8 on our superposition toy), and forcing it toward 1 destroys the
decomposition. Hence three dose regimes (§7): too weak = no effect; moderate = prunes needless
cancellation, which helped the LM on every metric; too strong on a superposed target = actively
destructive. Dose-find per target, watching the ratio and the all-on sanity check.

Optional extras, currently off in the main configurations: a **simplicity** penalty on each active
component's internal complexity (APD's nuclear norm — note it is *rotation-invariant*, so it cannot
do the basis-choosing job the L1 does), and a **lifetime** penalty (squared firing frequency) that
was our earlier granularity tool and has been superseded by the rank-1 structure on every target
where both were tried.

## 5. Training procedure and practical details

1. **Warmup**: before any gating, fit the components so they sum to the original weights (few
   hundred steps on faithfulness alone).
2. **Main loop**, per step: run the original model to get target outputs and activations; compute
   gates; apply the losses above; one optimizer step for components and gate network jointly.
3. **Spillover term**: during training (only), the gap between the component sum and the true
   weights rides along with its own random mask, absorbing what the components don't yet explain;
   at evaluation it is forced to zero so we always measure the components alone.
4. **Efficiency**: rank-1 pieces make everything cheap. The gated forward pass runs in the
   two-vector factored form without ever materializing per-input weight matrices, and the L1 uses
   the identity |a·bᵀ|₁ = |a|₁·|b|₁, so the whole method costs roughly the same as the SPD/VPD
   baseline at matched parameter budget. The interaction loss multiplies step cost ~3× via its
   extra deletion passes. Measured scale point: 4,096 components over all 24 matrices of a 14M
   model run at ~0.44 s/step on one H100 (34 GB) — capacity is not the bottleneck at this scale.
5. **Choose the component count C generously.** C must exceed the largest matrix rank for
   faithfulness to be possible at all, but the real requirement is larger: each component is a
   fixed network-wide combination, so C has to cover the number of *distinct mechanism
   combinations* the model uses. Too small a C forces every component to be a generalist that
   everything needs — the observable symptom is gates saturating "all on". (On the 14M model,
   C=512 collapsed all-on; C=4096 gave each token ~25 active components out of thousands of live
   rare specialists.)
6. **Save the best checkpoint, not the last** (implemented in the training loop, gated on the
   exact-summation sanity check so an unfaithful snapshot can never win). Reconstruction reliably
   peaks mid-training while the L1 keeps reorganizing the carving past the optimum — observed on
   every long run. A proper fix (annealing the L1 off, or stopping it at a target ratio) is still
   open; checkpoint selection is the current mitigation.
7. **Multi-GPU**: standard data-parallel training is wired in (identical initialization on all
   ranks, each rank on its own data shard, gradients averaged every step, adversary state kept
   per-rank). Launch with `torchrun`; the same scripts run single-process unchanged. Runs log to
   Weights & Biases when enabled (`WANDB=1`).

## 6. How we evaluate

- **Faithfulness suite**: all-components-on output vs the original (must be ~identical — the sanity
  check that the decomposition is even valid); gate-selected reconstruction (share of task
  performance recovered keeping only components the gate marks important); worst-case adversarial
  deletion damage; number of active components per position.
- **Ground-truth recovery** (toys only): with one known mechanism per input feature, does each
  feature get its own dedicated component (separation), do all features get one (coverage — never
  read separation without it: suppressing most features fakes perfect separation), does one
  component dominate per feature (purity), do components span the layers the mechanism spans, and
  does keeping *only* a feature's component preserve that feature's computation (sufficiency)?
- **Fingerprints** (models with meaningful units): each component's weight-mass distribution over
  heads/neurons/matrix-types/layers, and its gate profile over positions/timesteps.
- **Interaction matrix** (any model, no ground truth needed): pairwise joint-deletion damage,
  revealing redundant pairs (super-additive) and shared pathways (sub-additive).
- **L1 ratio** (any model): how far the carving is from disjoint weight support; also a regime
  diagnostic — a healthy carving near 1.0 indicates neuron-aligned structure, a healthy carving far
  above 1.0 indicates superposition, and that tells you where the L1 loss is safe to apply. Also
  computed per matrix, giving a map of *which parts* of a model are neuron-aligned before dosing.
- **Open-ended component interpretation** (language models): for each component, the contexts where
  it fires, the tokens it fires on, and — causally — which next-token predictions degrade when it
  alone is deleted. "Fires on the word *systems*; deleting it damages predictions of
  systems/machines/technology" is the shape of a positive result. Components are sampled two ways
  (most-used, and rare-but-strong), since those populations mean different things.

## 7. What is validated where — and the honest failure record

**Superposition toy (TMS, 5 features in 2 dims).** Near-perfect recovery of the known mechanisms
(mean best cosine similarity 0.982 vs the rank-1 baseline's 0.927). Single decomposed matrix, so it
does not test the whole-network gate.

**Cross-layer superposition toy (2-layer residual MLP, 100 features).** With rank-1 components and
*no* granularity tuning at all: separation ~0.90, coverage ~1.0, cross-layer 1.0, three seeds. The
matched VPD baseline plus its own published clustering method hits a wall on this toy: no clustering
strength gives per-feature separation and cross-layer structure at the same time (0.90 separation
forces 0.06 cross-layer). This is the method's strongest evidence.

**Handcrafted algorithmic RNN (AlgZoo 2nd-argmax, 726 params, full answer key, dense activity —
every input runs the whole algorithm).** The regime where all rate-based sparsity pressure is
uninformative. Final stack recovers the documented mechanisms including cross-timestep wiring, at
the exact disjoint-support optimum (ratio 1.00), gate-selected accuracy 0.86 vs the model's 0.99.
Residuals: the shared running-max machinery stays split across components (the objective is
provably indifferent there — pieces of an always-on mechanism have nothing to break the tie), and
"which granularity is correct" is genuinely ambiguous even with the answer key.

**Small real language model (attn-only-2l, attention-only).** Operating point (rank-1, C=512,
hidden-recon, L1 3e-3, faithfulness 1e8): 88.8% of language-modeling performance recovered keeping
only gate-selected components vs 81.8% for the matched-budget VPD baseline; exact summation (all-on
KL 0.009); component weight-mass concentrates on behaviorally-relevant heads at 2.5× uniform.
Deficits at this stage: worse worst-case robustness than VPD, low sparsity, functionally redundant
components. Single seed, short runs — superseded by the Pythia results below as the main LM
evidence.

**Pythia-14M (6 layers, MLPs included, Pile-trained — the first fully realistic target).** With
enough components (C=4096) and a training budget matched to the VPD baseline, the method **matches
the published baseline's sparsity and beats its reconstruction simultaneously**: ~83% CE-recovered
at ~0.5–0.7% components active per token (VPD: 82.2% at 0.7%), tighter exact-summation, replicated
across two seeds nearly digit-for-digit. A 5× budget extension improved reconstruction only ~7%
(best checkpoint kl 1.16), establishing that the ~82–86% recovery level is a property of the current
objective, not undertraining. **Interpretation** (open-ended, no target circuit): ~2,800 of 4,096
components are live somewhere; meaning stratifies by firing rate — a small always-on backbone of
generic-text machinery (large causal load, polysemantic); mid-rate components implementing
individual *syntax rules* (a "preposition → the" component is the largest single causal effect
found); rare components as *format and topic specialists* (a lexical component for the word
"systems" whose deletion specifically damages systems/machines/technology predictions; LaTeX-math,
academic-citation-markup, code-close-delimiter, and biomedical-topic components); and a residue of
near-dead duplicates. Interpretations are human glosses of top-token lists — automated labeling is
the missing rigor step.

**Failure record worth remembering:**
- Full-rank components blob or collapse on every dense target (LM and toys) regardless of penalty
  tuning; rank-1 is what fixed it, not a coefficient.
- Undercomplete C collapses the gates to all-on: too few whole-network components forces every one
  to be a generalist. Capacity is a first-class hyperparameter, not a budget knob.
- Rate-based granularity pressures (per-input minimality, firing-frequency penalties) are
  structurally blind on dense targets — mechanisms there differ by *role*, not by *how often* they
  run.
- The interaction loss achieved causal independence on the algorithmic RNN at zero accuracy cost,
  but **hurt LMs at every dose tried** without buying modularity; suspects are pair-sampling
  coverage (thousands of times sparser per pair than the RNN runs) and always-on gates giving
  pairwise deletion nothing input-specific to grip.
- The entrywise L1 has three regimes: inert (too weak), helpful (prunes redundant overlap — LMs
  benefited across all metrics at once at moderate dose), destructive (forcing disjointness onto
  genuinely superposed mechanisms — collapsed the residual-MLP toy at 1/1000th of the dose the RNN
  needed). It also **keeps aligning past the reconstruction optimum** on every long run — hence
  best-checkpoint saving, pending a real anneal. Dose-find on every new target; watch the ratio
  and the all-on sanity check while doing so.
- The adversarial-robustness gap (ours ~31 vs baseline ~3.6 on Pythia) survived every
  non-structural explanation tested: it is not the L1's alignment (same gap with L1 off), not
  training budget (unchanged by 5×), not seed noise (replicates). Remaining suspects are
  structural: one shared gate exposes all matrices at once, and thousands of dormant components
  that nothing trains to sum benignly in adversarial combinations.

## 8. Open problems

1. **Worst-case robustness** — the one axis where the per-matrix-atom baseline is clearly better
   (~10×), now known to be structural (see failure record). Candidate fixes: train dormant
   components against adversarial activation, or a stronger/longer adversary during training.
2. **The ~82–86% reconstruction ceiling** on real LMs — shared by the baseline method and
   confirmed budget-independent for ours; whatever lifts it is a new idea, not a bigger run.
3. **L1 scheduling**: anneal it off or stop at a target ratio instead of letting it grind past the
   reconstruction optimum; per-module dosing guided by the per-matrix ratio map (MLPs look
   neuron-aligned; attention subspaces don't).
4. **Causal modularity / redundancy**: the always-on backbone remains polysemantic and mutually
   redundant; the interaction loss is the right idea but needs an LM-workable form (restricted
   pair pools over the components that actually co-fire).
5. **Granularity underdetermination**: when a mechanism's pieces always fire together, the
   objective is indifferent to gluing them; needs a principled tie-breaker or an acceptance that
   several granularities are equally valid.
6. **Automated interpretation**: current component labels are human glosses of firing/ablation
   statistics; an LLM-judge labeling and scoring pass (as the VPD paper used) would make the
   monosemanticity claims quantitative.
7. **Scale**: largest target so far is 14M parameters. The published 67M benchmark model is wired
   up and, with data-parallel training, a full-budget run is days, not weeks.
