# Attribution-Gated Parameter Decomposition — Method (detailed)

*(nano_apd / `polar.py` (toys), `polar_lm.py` (LMs). Updated 2026-07-31. IG, Jacobian, rank-budget, and L0 experiments are recorded in §10; entropy was removed after its 67M regression. Targeted carving is documented separately in `carving_method.md`.)*

---

## 1. Goal

Decompose a frozen target network's weights into $C$ **parameter components** such that (a) the components sum back to the original weights exactly (*faithfulness*), (b) on any given input, a small input-dependent subset of components reproduces the model's computation (*sufficiency of the selected circuit*), and (c) individual components correspond to human-recognizable mechanisms (*interpretability*). No warm start from the target's weights, no trained mask network, no top-$k$, no stochastic mask sampling. The only learned objects are the component factors themselves; everything else (attribution, gating) is a deterministic differentiable function of them.

## 2. Parametrization

### 2.1 What gets decomposed

Every `nn.Linear` weight of the target's blocks — 24 matrices at both current scales:

- **pile-4L 67M** (VPD's benchmark target, 4-layer LlamaSimpleMLP, $d{=}768$): `h.{0..3}.attn.{q,k,v,o}_proj`, `h.{0..3}.mlp.{c_fc,down_proj}`.
- **Pythia-14M** ($d{=}128$): `gpt_neox.layers.{0..5}.attention.query_key_value`, `.attention.dense`, `.mlp.dense_h_to_4h`, `.mlp.dense_4h_to_h`.

Embeddings, unembeddings, layernorms, and biases are **not** decomposed; the gated model reuses the target's. The target is frozen throughout (`requires_grad` is enabled only so activation gradients can flow).

### 2.2 Factors

For each decomposed matrix $W^{(m)} \in \mathbb{R}^{d_{in} \times d_{out}}$ (note: stored transposed relative to `nn.Linear`'s `[d_out, d_in]`), a `ComponentBank` holds

$$A^{(m)} \in \mathbb{R}^{C \times d_{in} \times r},\qquad B^{(m)} \in \mathbb{R}^{C \times r \times d_{out}},\qquad W^{(m)} \;\stackrel{!}{=}\; \sum_{c=1}^{C} A^{(m)}_c B^{(m)}_c$$

with rank $r{=}4$ per matrix per component ($r{=}2$ costs ~2.5× faithfulness and concentration; effective-rank analysis shows ~2.8 of the 4 dims used). Xavier-normal init, trained from scratch with AdamW (wd 0), cosine LR.

**Component $c$** is the *collection* $\{A^{(m)}_c B^{(m)}_c\}_{m=1..24}$ — one piece in every matrix, tied together only by the shared index $c$ and the shared gate. This is what lets a component express a cross-layer mechanism.

### 2.3 What "rank 4" does and does not mean

Rank 4 is a **per-matrix** property. Each piece reads from a ≤4-dim subspace of its matrix's input and writes into a ≤4-dim subspace of its output. Across 24 matrices these subspaces are independent, so a component can touch up to $4 \times 24 = 96$ directions network-wide; stacked block-diagonally its pieces form a rank-≤96 object, not rank-4. An end-to-end "component rank" is not even well-defined — nonlinearities (softmax, SiLU/GELU, layernorm) sit between the pieces, and rank is a property of linear maps. The honest picture: a component is a circuit with 4-lane bottlenecks at each site. (Gating never changes rank: it scales pieces by a scalar.) Two coherent ways to impose a *global* rank-4, both unbuilt: shared per-component read/write bases against the residual stream, or a total effective-rank budget summed over matrices (§10.4's machinery). Measured on the trained 67M decomposition: components spread mass over ~14 of 24 matrices at the median (top-4 matrices hold 53%), so a global budget would be a real constraint, not a formalization of what training already finds.

## 3. Attribution — the exact computation

Attribution answers, per token and component: *how much does this component's write matter for the model's computation on this token?* It is recomputed fresh every training step from the current factors.

### 3.1 Capturing activations

The `Runner` monkeypatches every decomposed linear's `forward`. In **target mode** it computes `out = F.linear(x, W, b)` and caches $h^{(m)}_t = $ `pre` (the input) and `post` (the output) with graph intact. One forward of the frozen target over the batch `idx` `[B, T]` yields logits and the 24 cached pairs.

### 3.2 One backward for all gradients

The scalar objective is **summed** next-token cross-entropy (`reduction="sum"` — summed, not averaged, so per-token gradient magnitudes are independent of batch shape):

$$s = \sum_{b,t} \mathrm{CE}\big(\mathrm{logits}_{b,t},\; x_{b,t+1}\big)$$

A single `torch.autograd.grad(s, [post^{(1)}..post^{(24)}])` returns $g^{(m)}_t = \partial s / \partial\, \text{post}^{(m)}_t$ for every matrix, token, and batch element at once — **one backward** total. (The toys instead take per-output-dimension gradients — `grad_outputs=eye`, `is_grads_batched=True` — and sum squares over output dims, following APD's $\tfrac12\lVert y\rVert^2$ convention; that per-output convention would cost $V{=}50257$ backwards at LM scale, which is why the LM collapses it to the one summed-CE scalar.) Both $h$ and $g$ are **detached** after this: gates must be differentiable in the *components*, not in the target's pass.

### 3.3 The per-component inner product

Component $c$'s write at matrix $m$, token $t$ is $h^{(m)}_t A^{(m)}_c B^{(m)}_c \in \mathbb{R}^{d_{out}}$ — what this piece contributes to the matrix output. Its alignment with the loss gradient is the signed scalar

$$s_{t,c}^{(m)} = \big\langle g^{(m)}_t,\; h^{(m)}_t A^{(m)}_c B^{(m)}_c \big\rangle$$

which is exactly the **first-order Taylor estimate of the loss change from dialing this piece from present (1) to absent (0)** — grad × contribution. Implementation never materializes $A_cB_c$ (that would be $C \times d_{in} \times d_{out}$): it contracts through the rank bottleneck,

```
Bg    = einsum("cmo,bto->btcm", B, g)      # B_c g_t          [B,T,C,r]
inner = einsum("btd,cdm->btcm", h, A)      # h_t A_c          [B,T,C,r]
s     = (inner * Bg).sum(-1)               # ⟨g, h A_c B_c⟩   [B,T,C]
```

cost $\propto N \cdot C \cdot r \cdot (d_{in} + d_{out})$ per matrix — the dominant attribution cost, linear in $C$.

### 3.4 Aggregation across the 24 pieces: coherent (sum, then square)

$$\mathcal{A}_{t,c} = \Big(\sum_{m=1}^{24} s^{(m)}_{t,c}\Big)^{2}$$

The 24 signed scalars are summed **before** the single squaring — amplitude-like, not energy-like. Pieces of a cross-layer mechanism whose effects push the loss the same way accumulate coherently; pieces pushing in opposite directions cancel. The alternative ($\sum_m (s^{(m)})^2$) would credit a component for having *any* effect *anywhere*; the coherent form credits its **net effect as one object**, which is what makes the component — not its pieces — the unit of attribution. Known wrinkle: a component doing genuine push–pull across layers is under-credited (first order sees only the net).

### 3.5 Why squared (not raw, not $|\cdot|$)

The inner sum is signed; relevance is a magnitude (a strongly helpful and a strongly harmful component are both load-bearing). Squaring over $|\cdot|$: (i) smooth at 0 — the gate must be differentiable in the components, and training uses that path; (ii) contrast — squaring before max-normalization suppresses the sea of small noise credits (equivalent to $|\cdot|$ at $2\tau$, i.e. the choice is folded into the gate temperature and everything downstream is calibrated to squared-at-$\tau{=}1$); (iii) APD heritage (Gauss–Newton-like squared relevance).

### 3.6 Attribution gotchas

- **Final position**: the last token has no next-token label, so its CE gradient — and therefore every attribution and gate at that position — is identically 0. Any probe reading gates at "the answer position" must append a filler token and read at $-2$. (This silently zeroed an early probe.)
- Attribution scale varies by orders of magnitude across tokens (it tracks the CE gradient); nothing downstream may depend on absolute scale — hence the gate normalization (§4).

## 4. Gating

$$g_{t,c} = \Big(\frac{\mathcal{A}_{t,c}}{\max_{c'} \mathcal{A}_{t,c'} + \varepsilon}\Big)^{\tau} \in [0,1],\qquad \tau = 1$$

**Why divide by the per-token max** (and not the sum): two jobs — the gate must be a usable dial in $[0,1]$ for the gated pass, and it must be invariant to each token's overall attribution scale. Max-normalization does both while **preserving relative magnitudes**: the leading component always runs at exactly 1.0 and a component with 44% of the leader's attribution runs at 0.44. Sum-normalization would make gates *shares*: a token using 20 components would run each at ~0.05 and the gated network's signal would shrink with the number of participants — a degenerate coupling between "how many" and "how strong" that breaks reconstruction exactly when many mechanisms are needed. (The entropy loss of §10.2 deliberately uses the *sum*-normalized object instead — for measuring concentration you want the competitive, sums-to-1 quantity. Same attributions, two normalizations, two jobs.)

Properties: no fixed $k$ (each token activates however many components it earned); no batch coupling (APD's batch top-$k$ coupled samples); no learned gate network; **differentiable in the components** through $\mathcal{A}$, so the training gradient shapes *which* components claim which tokens, not just their contents. $\tau{>}1$ sharpens (toy $\tau{=}2$: separation 0.81 — worse; left at 1).

## 5. The gated forward

The second pass runs the same network with every decomposed matrix replaced by its **token-dependent** dial-weighted sum $\tilde W^{(m)}_t = \sum_c g_{t,c} A^{(m)}_c B^{(m)}_c$. Implementation never builds $\tilde W_t$ (that would be a $d_{in} \times d_{out}$ matrix *per token*): with tokens flattened to rows,

```
inner = einsum("nd,cdm->ncm", x, A) * g.unsqueeze(-1)   # gated rank-space acts
out   = inner.reshape(N, C*r) @ B.reshape(C*r, d_out)   # one GEMM
```

Biases (and all non-decomposed modules) are the target's own. This pass is the compute bottleneck (~85% of the backward). `--sub_frac 0.5` runs it on a random half of the batch's sequences each step (recon + act computed on that half; faithfulness/attribution/geometry still see everything).

## 6. Losses

Total, with $\rho(s) = \min(1, s/0.3S)$ ramping the functional terms over the first 30% of training (they fight faithfulness at random init; faith is full-strength from step 0):

$$\mathcal{L} = \lambda_f \mathcal{L}_{faith} + \rho(s)\big(\lambda_r \mathcal{L}_{recon} + \lambda_g \mathcal{L}_{geom} + \lambda_a \mathcal{L}_{act} + \lambda_{e} \mathcal{L}_{ent} + \lambda_{j} \mathcal{L}_{tan}\big)$$

Working weights, 67M: $\lambda_f{=}4{\times}10^4$, $\lambda_r{=}1$, $\lambda_g{=}30$, $\lambda_a{=}0.3$ (sweep winner); testing additions $\lambda_e{=}0.1$, $\lambda_j{=}1.0$ (§10).

### 6.1 Faithfulness

$$\mathcal{L}_{faith} = \frac{1}{N_{par}} \sum_m \Big\lVert \sum_c A^{(m)}_c B^{(m)}_c - W^{(m)}_{tgt} \Big\rVert_F^2$$

The weight-space anchor: the decomposition must *be* the model, not an imitation. Computed in fp32 outside the autocast region (§8). **Must be scale-calibrated per target**: what matters is faith relative to the target's mean squared weight, and the functional gradients it competes with scale with the loss, not the weights. Rule of thumb $\lambda_f \sim 100 \cdot \overline{W^2_{ref}} / \overline{W^2_{tgt}}$. The pile-4L failure that taught this: weight 100 (tuned on Pythia-14M) left faith at 100–450% of weight power on a target with 4× smaller weights — a decomposition that reconstructed behavior while not summing to the weights at all (Potemkin). Diagnostic for "is faith merely losing the tug-of-war": train with the functional losses off — if faith collapses to ~0 in a few hundred steps, the capacity is fine and the weight is wrong. At $\lambda_f{=}4{\times}10^4$: faith < 0.1% of weight power. **Always report faith as % of mean squared weight.**

### 6.2 Gated reconstruction

$$\mathcal{L}_{recon} = \mathrm{KL}\big(p_{tgt}(x_{t+1} \mid x_{\le t}) \,\big\Vert\, p_{gated}\big)$$

(`F.kl_div(log_softmax(gated), softmax(target.detach()), batchmean)`). This is the specialization pressure: suppressed components must be *dispensable*, kept components *sufficient*, at every token. Because gates are differentiable in the components, this loss simultaneously trains what components contain and which tokens they claim.

### 6.3 Usage-geometry matching (the "geometric loss")

**The target quantity.** For each token, the target model has a ground-truth answer to "which parameters did this token use": the per-parameter usage fingerprint

$$v_t = \mathrm{vec}\big[(h_t \otimes g_t) \odot W\big] \quad\text{concatenated over all 24 matrices} \;\in \mathbb{R}^{\#\text{params}}$$

— gradient × activation × weight, elementwise on every weight entry. Two tokens that run the same mechanism have $\cos(v_i, v_j) \approx 1$; tokens running different mechanisms $\approx 0$ (measured separation on toys: 0.99). This is a property of the *target alone* — no components involved — so it is a supervision signal for how usage should be organized.

**The loss.** Component-usage vectors $\mathcal{A}_i \in \mathbb{R}^C$ (rows of the attribution matrix) must reproduce that geometry:

$$\mathcal{L}_{geom} = \mathbb{E}_{(i,j) \sim \text{pairs}}\Big[\big( \cos(\mathcal{A}_i, \mathcal{A}_j) - \cos(v_i, v_j)\big)^2\Big]$$

This is the anti-merge pressure: a merged component that serves several mechanisms forces $\cos(\mathcal{A}_i, \mathcal{A}_j) \approx 1$ for different-mechanism pairs, which the target's $\cos(v_i, v_j) \approx 0$ exposes. 4096 token pairs sampled uniformly per step from the $B{\cdot}T$ flattened tokens.

**Never materializing $v$.** The fingerprints are ~14M–67M-dim; the loss needs only pairwise dots. For one matrix, with $P = h_i \odot h_j$ and $G = g_i \odot g_j$:

$$v_i \cdot v_j = \sum_{a,b} h_{i,a} g_{i,b} W_{ab} \cdot h_{j,a} g_{j,b} W_{ab} = P^\top \big(W^{\odot 2}\big)\, G$$

so exact pair dots come from one bilinear form against the **squared** weight matrix, summed over matrices (`einsum("pd,do,po->p", h_i*h_j, W², g_i*g_j)`), with the same identity giving the norms ($i{=}j$). Cost per step: negligible next to the gated pass.

**Why the absolute (MSE-on-cosine) form is load-bearing.** CLIP/InfoNCE-style replacements (softmax over similarity rows, cross-entropy against the target row) are invariant to per-row shifts and scale — they enforce *ranking within a row*, nothing more. The merged degenerate solution passes ranking: if everything routes through one giant component, cosines are 0.99 (related) vs 0.95 (unrelated) — ordering correct, contrastive loss satisfied, decomposition garbage. The anti-merge signal lives precisely in the absolute requirement that unrelated pairs sit at **zero**, which per-row invariance discards. Toy measurements: absolute form separation 0.90–0.98; CLIP-row 0.26 (coverage 0.5, purity 0.03 — collapsed); two-tower InfoNCE 0.67 (coverage 0.36 — the learned head absorbs the structure, leaving the usage vectors unconstrained). Functional score 73 → ~0. General lesson: the fingerprint geometry is a *calibrated target*, not a preference ordering.

### 6.4 Activation matching (read + write)

$$\mathcal{L}_{act} = \frac{1}{24} \sum_m \mathrm{MSE}\big( \text{post}^{(m)}_{gated},\; \text{post}^{(m)}_{tgt}.\mathrm{detach}()\big)$$

Anti-shortcut: output-matching alone is gameable — toy counterfeits produced correct outputs via the wrong layer (functional cross-layer 3–11/100 with perfect output match). Matching every decomposed matrix's output under the gated pass forces the gated model to take the target's *route*, not just reach its destination. (On toys this is explicitly split into read-side (post-ReLU hidden) and write-side (residual contribution) terms; at LM scale matching all 24 posts covers both sides.)

## 7. Optimization mechanics

- **Schedule**: cosine LR (1e-3 peak), AdamW wd 0, functional ramp 30%.
- **DDP**: the Runner's monkeypatch is incompatible with `DistributedDataParallel` wrapping, so gradient sync is a manual `all_reduce(AVG)` over bank grads after `backward()` — the loader shards by rank, each rank runs the full loss on its shard.
- **Checkpointing** (added after losing a run at step 99,900/100,000 to a late OOM): `--ckpt_every N` (production: 5000) writes banks + optimizer + step atomically (tmp, then `rename`) on rank 0; `--resume` restarts from it; a supervisor script auto-relaunches with `--resume` on crash, so a failure costs ≤ N steps. `torch.cuda.empty_cache()` every 2k steps caps allocator-fragmentation growth (~10GB over 90k steps was the killer).

## 8. Precision & efficiency

- **Surgical bf16** (`--bf16`): autocast covers the target pass, attribution, and gated pass; the faithfulness residual and master weights stay fp32, computed *outside* autocast. Zero measured faith cost. Blanket low precision (TF32 everywhere, or faith inside autocast) costs ~2.5× faith — precision noise lands on whichever pathway is unprotected. Cost of bf16: a functional ceiling at convergence (full-gate KL ~2× fp32's; fp32 recovers it at ~1.7× time) — acceptable for structure-finding runs, rerun fp32 if a functional-grade artifact is needed.
- **Batch subsampling** (`--sub_frac 0.5`): §5.
- Cost scales $\propto C \cdot r \cdot \sum_m (d_{in}{+}d_{out}) \cdot$ tokens. Measured: Pythia-14M ($C{=}1024$) ~5 H100-h; pile-4L 67M ($C{=}2048$) 0.24 s/step baseline → ~14 H100-h for 1.6B tokens — ~8× fewer tokens than the VPD reference decomposition of the same model, no CI network. Extensions: +0.10 s/step per extra IG point; jac ≈ +0.20 s/step at every-step application (§10.3's lazy mode cuts it ~4×).
- einsum ≤1.14× vs hand-tuned GEMM (not a bottleneck); evidence collection for auto-interp ≈ ¼ training cost per token (no gated pass).

## 9. Evaluation

**Toys (TMDR; ground truth total):** separation (does each true mechanism map to one component), coverage, keep-only error (run with only the assigned component: sufficiency), purity (mean attribution share of the top component), cross-layer consolidation, and the **functional layer-half referee** (oracle assignment + layer ablations) that catches counterfeits scoring 1.00 on weight-based metrics.

**LMs:** faith as % of weight power; held-out full-gate KL (nats/token); keep-top-$j$ KL curve (knee = per-token concentration); gates > 0.01 per token; behavior preservation (induction copy accuracy); auto-interp catalogs (streaming top-k evidence over held-out corpus positions — decoded with the **dataset's** tokenizer, which for pile-4L is the Pythia/NeoX tokenizer despite the GPT-2-sized vocab — LLM-labeled via the org API); causal edits (ablate labeled component sets vs random-set nulls); circuit correspondence against per-head ablation ground truth with a **specificity control** (per-head ablation scored on both the behavior *and* general Pile CE, to separate behavior-specific heads from shared infrastructure).

## 10. Extensions under test

All opt-in flags (default off) in both trainers; each validated on toys against the baseline before entering the LM trial.

### 10.1 IG-over-mask attribution (`--ig_steps K`)

Replace §3's single-point gradients with their average along the shared path that scales **all** decomposed weights by $\alpha_k = k/K$ (target pass runs with `weight * wscale`):

$$\mathcal{A}_{t,c} = \Big(\tfrac{1}{K} \sum_{k=1}^{K} \sum_m \big\langle g^{(m)}_t(\alpha_k),\; h^{(m)}_t(\alpha_k) A^{(m)}_c B^{(m)}_c \big\rangle\Big)^2$$

The counterfactual is "every gate off" ($\alpha{=}0$); for this linear parametrization a component's off→on write difference is exactly $h A_cB_c$, so no per-component passes are needed — one shared path prices all $C$ components in $K$ backwards. This is Aumann–Shapley credit with the completeness property (credits sum to the loss change from empty to full model), so saturated-but-necessary mechanisms — which the single-point sensor missed on the induction circuit (mixed heads L1H1/L1H5) — cannot be invisible. Chain-rule note: summing per-head *edge* credits over all downstream read sites with the same IG gradients reconstructs exactly this node-level number, so the scalar per-token gate **is** aggregated edge credit; explicit per-edge resolution stays an analysis tool (`edge_sensor.py`) until gates become pair-level. The $k{=}K$ pass is the ordinary clean pass, reused for recon/act/geom. Aggregation across matrices and the square are unchanged (§3.4–3.5).

Findings: toys tie baseline (sep 0.98/0.98, functional 81/83); on the existing 67M decomposition the IG ordering is *causally sharper* (ablating its top-128 induction-contrast components: copy acc 0.294 vs 0.419 for the trained sensor's ordering; random 0.715). Surprise: coarse grids ($K{=}2,3$) **beat** $K{=}8$ on that causal referee despite low rank agreement (top-64 overlap 12/64) — low-$\alpha$ gradients carry the induction-relevant credit. Agreement-with-fine-$K$ is not a valid metric; the causal referee is. IG gates are **not** drop-in at eval on a decomposition trained with the old sensor (gated copy acc → 0: magnitudes calibrated to the training gates); the upgrade pays only through (re)training.

### 10.2 Attribution-entropy per token *(REMOVED 2026-07-28)*

Was $\mathcal{L}_{ent} = \mathbb{E}_t[H(p_t)]$ on the sum-normalized shares $p_{t,c} = \mathcal{A}_{t,c}/\sum_{c'}\mathcal{A}_{t,c'}$. Toy dose–response was favorable (weight 0.1: purity .45→.70 at mild separation cost), but on the 67M run it correlated with a severe auto-interp regression (polysemantic+unclear 21%→55% of components, high-confidence labels 599→283) and it was dropped from all subsequent runs; the code has been removed from both trainers. Practically superseded by the per-token L0 pressure (§10.5), which targets the effective *count* rather than peakedness and validated cleanly at 14M.

### 10.3 Jacobian (tangent-space) matching (`--jac`, `--jac_beta`, `--jac_eps`, `--jac_every`)

Value-matching (recon, act) pins what the gated model computes; this pins the local **tangent map** — how perturbations propagate. Perturb the residual stream entering one randomly sampled block $\ell$ by $\varepsilon u$ ($u$ random unit vector per token; $\varepsilon$ = `jac_eps` × stream RMS, captured by the same hook) and match finite-difference responses at the logits:

$$\delta_{tgt} = \tfrac{F_{tgt}(h_\ell + \varepsilon u) - F_{tgt}(h_\ell)} {\varepsilon},\qquad \delta_{gated} = \tfrac{F_{gated}(h_\ell + \varepsilon u;\, g) - F_{gated}(h_\ell;\, g)}{\varepsilon}$$

$$\mathcal{L}_{tan} = \mathbb{E}_t\big[1 - \cos(\delta_{tgt}, \delta_{gated})\big] + \beta\, \mathbb{E}_t\Big[\log^2 \tfrac{\lVert \delta_{gated}\rVert + \epsilon}{\lVert \delta_{tgt}\rVert + \epsilon}\Big]$$

— direction and magnitude of propagation matched separately. **Gates are detached**: the question is "does the circuit selected for this token implement the target's local computation," not gate-selection dynamics. Finite differences, not exact JVP (differentiating a JVP w.r.t. the components introduces mixed second derivatives; FD needs only forward continuations). Cost: 3 extra forwards on firing steps — perturbed target (no-grad; cheap), gated base with detached gates (cannot reuse the recon pass: its graph has gates attached), perturbed gated. Both gated passes need graphs — each pass's gradient is $O(1/\varepsilon)$, only their *difference* is $O(1)$ and meaningful, so neither side can be value-detached. `--jac_every N` applies it lazily with weight × N on firing steps (StyleGAN2-style lazy regularization: same time-averaged pressure, ~1/N cost; the logging cadence is a multiple of N, so logged values are real). Toys (on top of 10.1): separation 0.99 (best of any arm), keep-only halved to 0.008, cross-layer 1.0.

### 10.4 batchAverageRank (`--rank_target`, `--rank_w`, cap via `--m/--rank_m`)

Variable per-component rank without a one-sided minimality pressure (those get broken through — cf. `--count`, Schatten gaming): raise the cap $r \to 8$ and constrain the **usage-weighted batch average** of effective rank to a target $\bar R$, two-sidedly:

$$\mathcal{L}_{rank} = \Big(\mathbb{E}_t\Big[\textstyle\sum_c \bar g_{t,c}\, \mathrm{effrank}(c) \big/ \sum_c \bar g_{t,c}\Big] - \bar R\Big)^2,\qquad \mathrm{effrank}(c) = \tfrac{(\sum_i \sigma_i)^2}{\sum_i \sigma_i^2}$$

$\sigma_i$ = singular values of $A_cB_c$ — implemented trace-only via the participation ratio $(\mathrm{tr}\,MN)^2/\mathrm{tr}(MN\cdot MN)$ with $M=A^\top A$, $N=BB^\top$ ($m{\times}m$, no eigendecomposition), mass-weighted across matrices, $\bar g$ **detached** (else the budget is satisfied by reshuffling gating instead of changing rank). Deviating in either direction costs, so there is no degenerate escape to zero. **Status**: implemented and validated. At 67M, rank_w sweep showed weight 30 needed (weight 1 was overpowered — rank drifted below cap); rank growth 2→4→8 works with zero-init B slices preserving faithfulness. At 14M (`p14_sum_l0_bal_bar4`) the budget held avg rank pinned at target 4.0 with cap 8 throughout — and never strained: rank ~4 appears adequate at this scale, making BAR a validated passenger there rather than an active ingredient.

### 10.5 Per-token L0 sparsity (`--l0_w`, optional `--l0_target`) and the ownership lesson

$\mathcal{L}_{L0} = \mathbb{E}_t\big[(\sum_c \mathcal{A}_{t,c})^2/\sum_c \mathcal{A}_{t,c}^2\big]$ — the participation ratio of pre-gating attribution: a smooth effective count of components serving each token, minimized (or budgeted two-sidedly via `--l0_target`). With `gate_norm=sum` at 14M: collapses $\mathrm{l0}_{eff}$ ~48→~5 at negligible recon cost, and ownership becomes real (winner shares 0.9+ vs 0.06–0.14 without). **The trade discovered**: unopposed minimization over-collapses to ~28–30 *coarse* owners (token-class catch-alls; fine features like a GCD-topic component destroyed) — the cheap escape is merging, not mechanism discovery. A usage load-balancing term ($C\sum_c \bar u_c^2$ on batch-marginal shares) fixed the ownership count (100 owners, 0% polysemantic owners, best-in-family recon 1.69) but was **rejected and removed 2026-07-28**: it is function-blind — it spreads usage statistically (splitting frequent token classes) and cannot sort rare mechanisms into their own components. The induction canary confirms: with or without any of these pressures, induction stays smeared (flat contrast spectrum, zero cleanly-ablatable carriers). Mechanism-level separation needs a function-aware pressure enforced *where the mechanisms occur* (see the modular-geometry `--geom_mode modular` machinery: sound loss after closing three gaming loopholes — global hinge on mined pairs against a fingerprint-mass-weighted softmin conjunction target — but null at 14M for coverage reasons: random pair mining over bulk text never samples mechanism-mode token pairs).

## 11. Results snapshot

- **TMDR toys**: separation 0.99–1.00, keep-only 0.006–0.03, functional 73–84/100.
- **Pythia-14M** ($C{=}1024$): faith ~1% of weight power, KL 0.039, 0 dead components, 241/1024 high-confidence labels.
- **pile-4L 67M** ($C{=}2048$, VPD's benchmark, run `full_s4f40k`): faith <0.1% of weight power, KL 0.077, induction 0.846 (target 0.833); 2048/2048 labeled (48M-position evidence), 599 high-confidence, 72 semantic-topic; ~21% polysemantic, stable under 6× evidence = real superposition. Circuit ownership: bigrams moderately owned (top-4 ablation −31%), paren-matching **fully owned** (top-16 → P=0 vs random null), induction **smeared** (128 components → 0.42 vs random 0.74). Dividing line: per-token-representable state localizes; cross-position content retrieval smears.
- Completed: `full_ig2_ent01_jacE4` (IG $K{=}2$ + entropy 0.1 + lazy jac). Its saved evaluation is diffuse (292 gates above 0.01/token; gated induction 0.686 vs target 0.833), and the entropy term was subsequently removed. The later `p4l_sum_jacE4` run must be evaluated with its saved sum-normalized gate recipe.

## 12. Known limitations (and their diagnosis)

1. **Per-token diffuseness**: ~180–292 gates > 0.01 per token; keep-top-$j$ has no knee. Entropy (§10.2) worsened 67M interpretations, while L0 (§10.5) produced coarse catch-alls. No validated concentration pressure currently fixes this at LM scale.
2. **Participation, not ownership**: any weight split summing to $W$ is equally faithful, and ensembles satisfy the functional losses, so nothing rewards a single component solely owning a circuit's directions (L2H2-QK ownership at 14M: top component 0.47% — chance; 67M L2H4: top component 4.14% — 6–17× controls, mild). Shared with VPD/MPD/mask methods. §10.4 and the weight-mass diffuseness measurement (§2.3) target this axis.
3. **The attribution sensor is first-order and node-local — with a measured, narrower-than-it-first-looked gap.** On the 67M induction circuit, the specificity control separates the four ablation-fatal heads into one induction-*specific* head (L2H4: copy → 0 at only +0.16 nats general CE), two *mixed* heads (L1H1/L1H5), and shared previous-token *infrastructure* (L0H3: +0.67 nats general CE — necessary but not specific). Every attribution tier finds L2H4; edge-level EAP-IG additionally recovers the mixed heads; L0H3's invisibility to contrast-based attribution is *correct* (it acts identically in both conditions). The honest residual gap — single-point node credit misses mixed mid-circuit heads — is what §10.1 addresses (in trial now); what "gate" means for pair-of-positions credit remains the open design fork.

## 13. Relation to prior methods

Original APD: gradient attribution + batch top-$k$ mask + faithfulness/recon/Schatten. We keep its attribution core and faithfulness; replace top-$k$ with continuous per-token dials (differentiable selection, no $k$, no batch coupling), add usage-geometry and read+write activation matching, drop unit-norm tricks. VPD: per-matrix rank-1 atoms + trained causal-importance net + stochastic mask sampling. We need no mask network and no mask sampling (one extra backward), at ~8× fewer training tokens on its own benchmark — with the same qualitative interp findings, including the same induction failure. MIB (EAP-IG-inputs): source of the counterfactual + IG + edge-credit ingredients adapted into §10.1, with the counterfactual moved from input-space to component-space (own-gate-at-zero) and task-contrast kept eval-side (training gates must retain infrastructure).

