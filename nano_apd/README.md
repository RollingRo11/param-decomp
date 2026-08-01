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


## Targeted parameter carving

`train_carving.py` is the low-cost research branch. It does **not** train a global
component dictionary. Given matched contexts and one selected next-token position, it
learns `C` rank-`m` pieces in a small set of matrices and defines the rest of each weight
as an implicit exact residual.

```bash
# Ground-truth recovery benchmark; automatically runs the C=1 rank-matched baseline
python -m nano_apd.run_carving_toy

# Main 410M run on both H100s (batch sizes are global, not per GPU)
torchrun --standalone --nproc_per_node=2 -m nano_apd.train_carving \
  --target hf --model_name EleutherAI/pythia-410m --tag induction_free \
  --C 16 --rank_m 4 --auto_modules 8 --batch_pairs 16 --batch_retain 32 --bf16

# Required parameter-matched editing baseline: one component with the same total rank
torchrun --standalone --nproc_per_node=2 -m nano_apd.train_carving \
  --target hf --model_name EleutherAI/pythia-410m --tag induction_rank64 \
  --C 1 --rank_m 64 --auto_modules 8 --batch_pairs 16 --batch_retain 32 --bf16

# Stricter control: each piece is a projection of the existing target weight
torchrun --standalone --nproc_per_node=2 -m nano_apd.train_carving \
  --target hf --model_name EleutherAI/pythia-410m --tag induction_projected \
  --bank_type projected --C 16 --rank_m 4 --auto_modules 8 \
  --batch_pairs 16 --batch_retain 32 --bf16

# Independent held-out evaluation, including a low-rank relearning attack
python -m nano_apd.eval_carving --run pythia-410m_carving_induction_free \
  --relearn_steps 100

# Re-evaluate an older targeted.py artifact with checked-in code
python -m nano_apd.eval_targeted --run pile4l_targeted_v1
```

The built-in induction contrast changes one earlier token while keeping the selected cue,
target token, and local context identical. Training and localization ignore examples
where the target misses the selected positive token unless `--include_target_incorrect`
is explicitly set. A custom dataset can be supplied with
`--pairs_file FILE.pt`; it must contain `[N,T]` `positive` and `negative` token tensors,
`[N]` prediction `positions`, and optionally `[N]` `labels`. A retain tensor can be
supplied with `--retain_file` for offline/reproducible training. The default retain
stream is the repo's pre-tokenized Pile stream and is sharded across DDP ranks; for a
non-Pythia Hugging Face model, provide token IDs produced by that model's tokenizer.

The carving claim is intentionally narrower than full parameter decomposition. A run is
only evidence for multiple parameter components if individual pieces have distinct causal
effects and unseen subsets compose predictably. The mandatory comparison for `C × rank_m`
is a single component of rank `C * rank_m`; without a win over that baseline the bank is a
factorized LoRA edit, not an overcomplete mechanistic decomposition. See
[`carving_method.md`](carving_method.md) for the objective and decision criteria and
[`pilot_410m.md`](pilot_410m.md) for the deliberately negative 20-update engineering
pilot. The evaluator follows the saved BF16 setting by default; pass `--precision fp32`
for a robustness check, which is written to a separate report.

### Evaluation correctness fixes

`eval_lm.py` now uses the gate normalization saved in each run, infers the target from its
config, and refuses a silent train-data fallback. `polar_lm.py` reports reconstruction KL
in nats per predicting token and clears non-optimizer target gradients. `targeted.py`
checkpoints optimizer state, supports `--resume`, and records the actual probe split.


## Cross-layer induction components

`train_induction_components.py` trains 8, 16, or 32 task-conditioned component indices across every transformer layer, using sum-normalized attribution gates, an implicit exact residual, and stochastic intact/routed ablations. `eval_induction_roles.py` tests controlled functional variants; `eval_induction_natural.py` checks individual components on held-out Pile text; and `eval_induction_components.py --ground_truth` compares target head attribution with direct head ablation. See [`INDUCTION_COMPONENT_RESULTS.md`](INDUCTION_COMPONENT_RESULTS.md) for the mechanism, commands, complete C=8/16/32 results, negative results, and next-step criteria.

## Token-conditioned weight carving

`run_token_weight_carving.py` is the optimizer-free follow-up. It computes a factorized
selected-token contrast gradient, extracts four fixed rank-one directions from every
Pythia-410M transformer matrix, ranks them with sum-normalized attribution, and evaluates
a frozen 16-piece shortlist with exact ablation, FP32 path integration, stochastic
coalitions, prompt controls, and a matched direct-head reference. Ordinary and
diagonal-KFAC geometries are implemented as independent controls.

This is not an APD dictionary: each candidate is local to one matrix, and the intact
remainder of the model is implicit. The corrected runs required no optimizer steps and
about 5.2 GiB per H100. Their nine supporting pieces jointly removed 60--68% of the
held-out selected margin, but the pieces had highly redundant behavioral fingerprints
and did not recover the positive induction-head layer profile. See
[`TOKEN_WEIGHT_CARVING_RESULTS.md`](TOKEN_WEIGHT_CARVING_RESULTS.md) for the complete
results, explicit negative findings, commands, and checked-in artifacts.
