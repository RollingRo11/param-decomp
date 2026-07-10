# Matryoshka parameter decomposition — method

## The idea

Normal VPD breaks a model's weights into many tiny **atoms** (rank-1 pieces, one per output
direction of each weight matrix) and learns one on/off gate per atom per input.

Matryoshka adds a layer on top: instead of gating atoms one by one, we group them into a fixed
number of **components**. A component is a bundle of atoms that can come from *any* layer
(cross-layer). The network learns both:

1. **which atoms belong to which component** (a learned membership table), and
2. **how on each component should be** for a given input (one gate per component).

Goal: discover reusable, cross-layer structure *during* training, instead of decomposing into atoms
first and clustering them afterward.

## How it works

- The weights are still split into `A` atoms (same count as normal VPD, so faithfulness is
  unchanged). `A = 7104` for the 2-layer SimpleStories model.
- A learned **membership table** `M` has shape `[A, G]`. Entry `M[a, g]` answers "is atom `a` part
  of component `g`?" There are two ways to form it (`membership_type`):
  - **`sigmoid`** (original): each gate independent in `[0,1]`, so an atom can be in many components,
    one, or none — components **share** atoms. Downside: see "Committing the membership" below.
  - **`softmax`** (per-atom row sums to 1): each atom commits to ~one component. No sharing, but it
    *forces* atoms to spread across components — which the sigmoid version fails to do.
- A small **CI transformer** reads the input and outputs one importance gate per component:
  `[B, S, G]`.
- We turn component gates into per-atom gates with a **weighted average**:

  ```
  atom_mask[a] = ( Σ_g M[a,g] · component_gate[g] ) / ( Σ_g M[a,g] + 1e-6 )
  ```

  The denominator keeps every atom mask in `[0, 1]` no matter how many components an atom joins —
  avoiding saturation and blow-up.
- A **temperature** `tau` controls how sharply the membership logits map into gates. Historically it
  was annealed high→low to commit the gates over training. We now hold it **constant** and let the
  commitment *loss* (below) do the committing — annealing tau low is what caused the LM collapse
  (see `README.md`), and the loss does the same job at a safe temperature.

## Committing the membership (the hard part)

The single biggest issue with the method. The weighted-average aggregation is **scale-invariant in
`M`** — the `Σ M` in the denominator cancels, so reconstruction depends only on the *relative*
membership, never its magnitude. That means **nothing in the reconstruction objective pushes
membership to commit.** Two consequences we verified on toy models (46 configs):

- With **`sigmoid`** membership, the sparsity penalties (CM, CCS) can only *shrink* membership.
  There is no force toward "many small, distinct components", so every setting lands on either a few
  **mega-components** (one component grabbing ~all atoms) or **total collapse** (everything→0). No
  coefficient tuning escapes this — it is a hard cliff, not a tuning miss.
- The only thing that commits `sigmoid` gates is tau→0, which conflicts with CM (high gain at low
  tau) and reproduces the LM collapse.

The fix is **structural, not a coefficient**:

- **`softmax`** membership makes each atom's row sum to 1, so an atom *cannot* pile into one giant
  component — and reconstruction then forces atoms that need independent gating into *different*
  components. This distributes by construction (toy: ~60+ small components vs sigmoid's 1–2 blobs).
- A **commitment loss** (`coeff_assign_entropy`) pushes gates to commit *independently of tau*:
  - `sigmoid`: minimize **binary entropy** `−M·logM − (1−M)·log(1−M)` per gate → each gate to {0,1}.
  - `softmax`: minimize **row entropy** `−Σ_g M·logM` per atom → each atom one-hot.
  Ramp its coefficient up over training (`ae_ramp_frac`) so the assignment explores early, commits
  late — the schedule tau-annealing used to provide.

Note the penalties change meaning under `softmax`: CM (`mean M`) becomes the constant `1/G` (useless,
turn it off), and CCS (`component_size_l2`) becomes a **load-balancing** term that spreads atoms
across components.

## The losses

Four carried over from VPD, two new for matryoshka:

| Loss | Coeff | What it does |
|---|---|---|
| Faithfulness | `coeff_faith = 1e7` | atoms must rebuild the original weights (`‖W − VU‖²`) |
| Importance-minimality | `coeff_imp = 0.001`* | components should be off when not needed (sparse usage) |
| Stochastic recon | `coeff_stoch = 0.5` | output matches target under random gating (KL) |
| Persistent-PGD recon | `coeff_ppgd = 0.5` | same, but under an adversarial worst-case gating (KL) |
| **Membership L1** | `coeff_membership` (**CM**) | `sigmoid`: tax on total membership mass → sparse components. `softmax`: constant `1/G`, **useless — turn off** |
| **Component size** | `coeff_comp_size` (**CCS**) | `sigmoid`: quadratic per-component size tax. `softmax`: **load-balancing** (spreads atoms across components) |
| **Commitment** | `coeff_assign_entropy` (**AE**) | commits the membership, tau-independent. `sigmoid`: binary entropy → gates to {0,1}. `softmax`: row entropy → each atom one-hot. Ramped over training (`ae_ramp_frac`) |

\* `coeff_imp` defaults to `2e-4` in the config; the SS-2L runs use `0.001`.

Under `sigmoid` membership, **CM/CCS can only shrink membership** and never produce distributed
components (see "Committing the membership"). Under `softmax`, **AE** (row-entropy commitment) is the
key knob and CCS does the load-balancing; CM is off.

## Hyperparameters

Matryoshka-specific (the ones we sweep):

| Name | Default | Meaning |
|---|---|---|
| `n_components` (G) | 1024 | total number of cross-layer components |
| `membership_type` | `sigmoid` | `sigmoid` (shareable gates) or `softmax` (per-atom row sums to 1, distributes). **`softmax` is the current direction** |
| `coeff_membership` (CM) | 0.001 | small/sparse-component tax (`sigmoid` only — useless under `softmax`) |
| `coeff_comp_size` (CCS) | 0.0 | component-size tax (`sigmoid`) / load-balancing (`softmax`) |
| `coeff_assign_entropy` (AE) | 0.01 | commitment pressure (binary entropy for `sigmoid`, row entropy for `softmax`) |
| `ae_ramp_frac` | 0.6 | ramp AE 0→full over this fraction of steps (explore early, commit late) |
| `tau_start` / `tau_end` | 2.0 / 0.5 | membership temperature. Now held **constant** (set both equal); commitment comes from AE, not annealing. Annealing tau→low caused the LM collapse |
| `tau_anneal_frac` | 0.6 | (legacy) fraction of `n_steps` to anneal tau over before holding |
| `m_logits_init_std` | 1.0 | init spread of the membership logits |
| `m_logits_init_bias` | 0.0 | init offset; 0 = warm start (~half gates on) |

Shared training knobs (SS-2L comparison values in parentheses where they differ from defaults):

| Name | Default | Meaning |
|---|---|---|
| `n_steps` | 400000 (4000) | training steps |
| `batch_size` | 64 (32) | sequences per step |
| `seq_len` | 512 (256) | tokens per sequence |
| `main_lr` | 5e-5 (3e-4) | learning rate for atoms + CI net + membership |
| `main_lr_final_frac` | 0.1 | final LR as a fraction of `main_lr` (cosine decay) |
| `faithfulness_warmup_steps` | 400 (200) | warmup steps fitting atoms to weights only |
| `faithfulness_warmup_lr` | 1e-3 | LR during warmup |
| `coeff_faith` | 1e7 | faithfulness weight |
| `coeff_imp` | 2e-4 (0.001) | importance-minimality weight |
| `coeff_stoch` | 0.5 | stochastic recon weight |
| `coeff_ppgd` | 0.5 | adversarial recon weight |
| `p_start` / `p_end` | 2.0 / 0.4 | minimality norm exponent, annealed over training |
| `imp_beta` | 0.5 | shape term in the minimality loss |
| `leaky_alpha` | 0.01 | leak in the gate nonlinearity |
| `grad_clip_components` | 0.01 | grad clip on atom params |

CI transformer (the net that emits component gates):

| Name | Default (SS-2L) | Meaning |
|---|---|---|
| `ci_d_model` | 2048 (512) | hidden width |
| `ci_n_blocks` | 8 (4) | transformer layers |
| `ci_n_heads` | 16 (8) | attention heads |
| `ci_mlp_hidden` | 8192 (2048) | MLP width |
| `ci_rope_base` | 10000.0 | RoPE base |

Persistent-PGD (the adversarial gating):

| Name | Default | Meaning |
|---|---|---|
| `ppgd_lr` | 0.01 | step size for the adversarial gates |
| `ppgd_inner_steps` | 2 | adversary updates per training step |
| `ppgd_warmup_pct` | 0.025 | warmup fraction for the adversary LR |
| `ppgd_beta1` / `ppgd_beta2` | 0.5 / 0.99 | Adam moments for the adversary |

## Training notes

- **Save the best checkpoint, not the last.** `matryoshka.decompose` saves `<save_path>.best.pt`
  whenever eval `kl_ci_masked` improves, in addition to the final-step `<save_path>.pt`. The
  final-step model can be a *post-collapse corpse* — the LM run's best decomposition was mid-training
  and beat VPD, while its final step looked terrible. Always evaluate the `.best.pt`.
- **Normalize the reconstruction loss on small targets.** On toy models the output is small/sparse,
  so the raw MSE recon is tiny (~0.002) and the minimality penalty dwarfs it — driving *all*
  importances to zero and coasting on the non-decomposed passthrough. The toy harness divides recon
  by the target's variance so it is O(1). (LM uses KL, already O(1), so this is toy-specific.)

## Status

The `softmax` membership + ramped commitment loss is the current direction: it solves the
distribution problem that no `sigmoid` coefficient setting could (verified on the cross-layer
residual-MLP toy model with known ground truth). Reconstruction-vs-commitment balance and transfer
back to the LM are still being validated. See `README.md` for the full VPD-vs-MPD story and results.
