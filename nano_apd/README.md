# nano_apd

Minimal, faithful re-implementation of **Attribution-based Parameter Decomposition** (APD,
arXiv 2501.14926), mirroring the reference code at github.com/ApolloResearch/apd (cloned at
`/workspace/apd-reference`; the package is named `spd` there for historical reasons). The
purpose is a small, fast-to-iterate codebase in which we can swap the attribution method
for newer attribution/gradient techniques and measure the effect, primarily on the **toy
model of cross-layer distributed representations**.

## Method (as in the paper)

Each decomposed weight matrix W (per layer) is written as a sum of C parameter components
W = Σ_c A_c B_c (A_c: [d_in, m], B_c: [m, d_out]); component c is shared across layers
(one index c spans the whole network). Per batch:

1. **Attribution**: A_c(x) = Σ_o (Σ_layers ∂y_o/∂h_layer · a_c,layer(x))², where y is the
   *target model* output, h_layer its post-weight activations, and a_c,layer(x) =
   x_layer.detach() @ (A_c B_c) the component's output at that layer.
2. **Batch top-k**: keep the (topk × batch_size) highest-attribution (sample, component)
   pairs per instance; forward again with only those components active.
3. **Losses**: faithfulness (MSE between Σ_c A_cB_c and target weights, per-parameter),
   top-k reconstruction (match target output on the sparse forward pass), activation
   reconstruction (resid-mlp only: match post-ReLU hidden activations), and a Schatten-p
   norm penalty on the active components (simplicity/minimum description length).

## Files

- `apd.py` — components (A/B factorization, tied transpose), gradient attribution
  (vectorized via `is_grads_batched`, with an exact loop fallback), batch top-k, all
  losses, `optimize()` loop. Each function names its counterpart in the reference.
- `models.py` — TMS and residual-MLP target + APD models, sparse-feature dataset, target
  training (reference hyperparameters).
- `run_tms.py` — TMS 5-2 and 40-10 experiments (paper configs), MMCS/ML2R eval, polygon plots.
- `run_resid.py` — 1-layer (compressed computation) and 2-layer (**cross-layer distributed
  representations**) residual MLP experiments, neuron-contribution eval + plots.
- `tests.py` — numerical cross-checks of attribution / top-k / Schatten against the
  reference implementation, plus model consistency checks.

## Running

```bash
source /workspace/param-decomp/.venv/bin/activate
python -m nano_apd.tests                                     # cross-checks vs reference
CUDA_VISIBLE_DEVICES=0 python -m nano_apd.run_tms --variant 5-2
CUDA_VISIBLE_DEVICES=0 python -m nano_apd.run_tms --variant 40-10
CUDA_VISIBLE_DEVICES=1 python -m nano_apd.run_resid --n_layers 2
```

Targets are cached in `nano_apd/out/targets/`; each run writes `summary.json`,
`metrics_log.json`, checkpoints and plots to `nano_apd/out/<run_name>/`.

## Evaluation

- **TMS**: MMCS and ML2R between learned components and ground-truth mechanisms (the
  per-feature rank-1 slices of the target weights); polygon plots for the 5-2 variant.
- **Resid-MLP**: per-feature neuron-contribution vectors (`diag_relu_conns`, as in the
  reference plotting code): cosine similarity between each feature's target contribution
  vector and its best-matching component (`mmcs_conns`), plus component classification
  (dead / monosemantic / duosemantic / polysemantic at the reference cutoff 4e-2).

## Known deviations from the reference

None intended in the math. Structural differences only: no hook framework (forwards
return explicit caches), no wandb, no n>0 hidden-layer TMS variant, and the attribution
sum over output indices is computed with one batched `autograd.grad` call instead of a
Python loop (verified identical in `tests.py`).
