"""Minimal faithful re-implementation of Attribution-based Parameter Decomposition (APD).

Mirrors the reference implementation at github.com/ApolloResearch/apd (package `spd`,
cloned at /workspace/apd-reference) — same math, same hyperparameter semantics — without
the hook framework. Every function below has a named counterpart in the reference:

    LinearComponent            <- spd/models/components.py:LinearComponent
    calc_grad_attributions     <- spd/utils.py:calc_grad_attributions
    calc_topk_mask             <- spd/utils.py:calc_topk_mask
    calc_param_match_loss      <- spd/run_spd.py:calc_param_match_loss
    calc_schatten_loss         <- spd/run_spd.py:calc_schatten_loss
    calc_act_recon             <- spd/run_spd.py:calc_act_recon
    optimize                   <- spd/run_spd.py:optimize

Conventions (kept from the reference): all tensors carry an explicit n_instances
dimension; weights are [n_instances, d_in, d_out]; components are W_c = A_c @ B_c with
A [n_instances, C, d_in, m], B [n_instances, C, m, d_out]. A model's forward returns
(out, cache) where cache[param_name] holds "pre" (input to the weight), "post" (output of
the weight, before any bias) and, for APD models, "component_acts" (per-component output
before summing over C).
"""

import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path

import einops
import torch
from torch import Tensor, nn


def init_param_(param: Tensor, scale: float = 1.0, init_type: str = "kaiming_uniform") -> None:
    if init_type == "kaiming_uniform":
        torch.nn.init.kaiming_uniform_(param)
        with torch.no_grad():
            param.mul_(scale)
    elif init_type == "xavier_normal":
        torch.nn.init.xavier_normal_(param, gain=scale)
    else:
        raise ValueError(init_type)


class Linear(nn.Module):
    """Target-model linear layer, weight [n_instances, d_in, d_out]."""

    def __init__(self, d_in: int, d_out: int, n_instances: int, init_type: str, init_scale: float = 1.0):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(n_instances, d_in, d_out))
        init_param_(self.weight, scale=init_scale, init_type=init_type)

    def forward(self, x: Tensor, cache: dict | None = None, name: str = "") -> Tensor:
        out = einops.einsum(x, self.weight, "batch i d_in, i d_in d_out -> batch i d_out")
        if cache is not None:
            cache[name] = {"pre": x, "post": out}
        return out


class LinearComponent(nn.Module):
    """APD linear layer: W = sum_c A_c B_c. A [i, C, d_in, m], B [i, C, m, d_out]."""

    def __init__(self, d_in: int, d_out: int, C: int, m: int, n_instances: int,
                 init_type: str, init_scale: float = 1.0):
        super().__init__()
        self.C = C
        self.m = m
        self.A = nn.Parameter(torch.empty(n_instances, C, d_in, m))
        self.B = nn.Parameter(torch.empty(n_instances, C, m, d_out))
        init_param_(self.A, scale=init_scale, init_type=init_type)
        init_param_(self.B, scale=init_scale, init_type=init_type)

    @property
    def component_weights(self) -> Tensor:  # [i, C, d_in, d_out]
        return einops.einsum(self.A, self.B, "i C d_in m, i C m d_out -> i C d_in d_out")

    @property
    def weight(self) -> Tensor:  # [i, d_in, d_out]
        return einops.einsum(self.A, self.B, "i C d_in m, i C m d_out -> i d_in d_out")

    def forward(self, x: Tensor, topk_mask: Tensor | None = None,
                cache: dict | None = None, name: str = "") -> Tensor:
        inner_acts = einops.einsum(x, self.A, "batch i d_in, i C d_in m -> batch i C m")
        if topk_mask is not None:
            inner_acts = einops.einsum(inner_acts, topk_mask, "batch i C m, batch i C -> batch i C m")
        component_acts = einops.einsum(inner_acts, self.B, "batch i C m, i C m d_out -> batch i C d_out")
        out = einops.einsum(component_acts, "batch i C d_out -> batch i d_out")
        if cache is not None:
            cache[name] = {"pre": x, "post": out, "component_acts": component_acts}
        return out


class TransposedLinearComponent(nn.Module):
    """Tied transpose of a LinearComponent (used for TMS's W^T unembedding).

    A' = B^T, B' = A^T, so W' = (A B)^T. No parameters of its own.
    """

    def __init__(self, original: LinearComponent):
        super().__init__()
        self.original = [original]  # list to avoid registering as a submodule twice
        self.C = original.C
        self.m = original.m

    @property
    def A(self) -> Tensor:
        return einops.rearrange(self.original[0].B, "i C m d_out -> i C d_out m")

    @property
    def B(self) -> Tensor:
        return einops.rearrange(self.original[0].A, "i C d_in m -> i C m d_in")

    @property
    def component_weights(self) -> Tensor:
        return einops.einsum(self.A, self.B, "i C d_in m, i C m d_out -> i C d_in d_out")

    @property
    def weight(self) -> Tensor:
        return einops.einsum(self.A, self.B, "i C d_in m, i C m d_out -> i d_in d_out")

    def forward(self, x: Tensor, topk_mask: Tensor | None = None,
                cache: dict | None = None, name: str = "") -> Tensor:
        inner_acts = einops.einsum(x, self.A, "batch i d_in, i C d_in m -> batch i C m")
        if topk_mask is not None:
            inner_acts = einops.einsum(inner_acts, topk_mask, "batch i C m, batch i C -> batch i C m")
        component_acts = einops.einsum(inner_acts, self.B, "batch i C m, i C m d_out -> batch i C d_out")
        out = einops.einsum(component_acts, "batch i C d_out -> batch i d_out")
        if cache is not None:
            cache[name] = {"pre": x, "post": out, "component_acts": component_acts}
        return out


# ---------------------------------------------------------------------------
# Attributions
# ---------------------------------------------------------------------------

def calc_grad_attributions(
    target_out: Tensor,             # [batch, i, d_out_model]
    pre_weight_acts: dict[str, Tensor],
    post_weight_acts: dict[str, Tensor],
    component_weights: dict[str, Tensor],
    C: int,
    vectorized: bool = True,
) -> Tensor:                        # [batch, i, C]
    """Gradient attribution: A_c(x) = sum_o ( sum_layers dy_o/dh_layer . (x_layer.detach() @ W_c) )^2.

    Gradients are taken through the *target* model's graph w.r.t. its post-weight
    activations; component_acts use detached target pre-weight activations times the APD
    component weights (identical to the reference).
    """
    assert set(pre_weight_acts) == set(post_weight_acts) == set(component_weights)
    names = list(post_weight_acts.keys())

    component_acts = {
        name: einops.einsum(
            pre_weight_acts[name].detach().clone(), component_weights[name],
            "batch i d_in, i C d_in d_out -> batch i C d_out",
        )
        for name in names
    }
    out_dim = target_out.shape[-1]
    post_list = [post_weight_acts[n] for n in names]

    if vectorized:
        # Identical to the per-output-index loop below: grad of sum_batch y[..., o] for
        # each o, batched via is_grads_batched (vmap over the identity basis).
        eye = torch.eye(out_dim, device=target_out.device, dtype=target_out.dtype)
        summed_out = einops.einsum(target_out, "batch i d_out -> d_out")
        grads = torch.autograd.grad(
            summed_out, post_list,
            grad_outputs=eye, is_grads_batched=True, retain_graph=True,
        )  # each: [d_out_model, batch, i, d_out_layer]
        feature_attributions = torch.zeros(
            (out_dim,) + target_out.shape[:-1] + (C,), device=target_out.device, dtype=target_out.dtype
        )
        for grad, name in zip(grads, names, strict=True):
            feature_attributions += einops.einsum(
                grad, component_acts[name], "o batch i d_out, batch i C d_out -> o batch i C"
            )
        attribution_scores = einops.einsum(feature_attributions**2, "o batch i C -> batch i C")
        return attribution_scores

    attribution_scores = torch.zeros(
        target_out.shape[:-1] + (C,), device=target_out.device, dtype=target_out.dtype
    )
    for feature_idx in range(out_dim):
        feature_attributions = torch.zeros_like(attribution_scores)
        grad_post = torch.autograd.grad(
            target_out[..., feature_idx].sum(), post_list, retain_graph=True
        )
        for grad, name in zip(grad_post, names, strict=True):
            feature_attributions += einops.einsum(
                grad, component_acts[name], "batch i d_out, batch i C d_out -> batch i C"
            )
        attribution_scores += feature_attributions**2
    return attribution_scores


def calc_activation_attributions(component_acts: dict[str, Tensor]) -> Tensor:
    """L2^2 of each component's output, summed over layers. [batch, i, C]."""
    first = next(iter(component_acts.values()))
    scores = torch.zeros(first.shape[:-1], device=first.device, dtype=first.dtype)
    for acts in component_acts.values():
        scores += acts.pow(2).sum(dim=-1)
    return scores


def calc_topk_mask(attribution_scores: Tensor, topk: float, batch_topk: bool) -> Tensor:
    """[batch, i, C] -> bool mask. batch_topk selects int(topk*batch) over the flattened
    (batch, C) dims per instance."""
    batch_size = attribution_scores.shape[0]
    k = int(topk * batch_size) if batch_topk else int(topk)
    if batch_topk:
        scores = einops.rearrange(attribution_scores, "b i C -> i (b C)")
        topk_indices = scores.topk(k, dim=-1).indices
        mask = torch.zeros_like(scores, dtype=torch.bool)
        mask.scatter_(dim=-1, index=topk_indices, value=True)
        return einops.rearrange(mask, "i (b C) -> b i C", b=batch_size)
    topk_indices = attribution_scores.topk(k, dim=-1).indices
    mask = torch.zeros_like(attribution_scores, dtype=torch.bool)
    mask.scatter_(dim=-1, index=topk_indices, value=True)
    return mask


# ---------------------------------------------------------------------------
# Losses
# ---------------------------------------------------------------------------

def calc_recon_mse(output: Tensor, labels: Tensor) -> Tensor:
    """[batch, i, d] -> [i]."""
    return einops.reduce((output - labels) ** 2, "b i f -> i", "mean")


def calc_param_match_loss(target_weights: dict[str, Tensor], apd_weights: dict[str, Tensor],
                          n_params: int) -> Tensor:
    """Faithfulness: MSE between summed components and target weights, / n_params. -> [i]."""
    loss = 0.0
    for name in target_weights:
        loss = loss + ((apd_weights[name] - target_weights[name]) ** 2).sum(dim=(-2, -1))
    return loss / n_params


def calc_schatten_loss(As: dict[str, Tensor], Bs: dict[str, Tensor], mask: Tensor,
                       p: float, n_params: int) -> Tensor:
    """Schatten-p penalty on the masked components (simplicity loss). -> [i].

    mask is the bool topk mask [batch, i, C] (or float attributions in the Lp case).
    """
    batch_size = mask.shape[0]
    schatten_penalty = None
    for name in As:
        A, B = As[name], Bs[name]
        S_A = einops.einsum(A, A, "i C d_in m, i C d_in m -> i C m")
        S_B = einops.einsum(B, B, "i C m d_out, i C m d_out -> i C m")
        S_AB = S_A * S_B
        S_AB_topk = einops.einsum(S_AB, mask.to(S_AB.dtype), "i C m, batch i C -> batch i C m")
        term = ((S_AB_topk + 1e-16) ** (0.5 * p)).sum(dim=(0, -2, -1))  # -> [i]
        schatten_penalty = term if schatten_penalty is None else schatten_penalty + term
    return schatten_penalty / n_params / batch_size


def calc_act_recon(target_post_acts: dict[str, Tensor], layer_acts: dict[str, Tensor]) -> Tensor:
    """MSE between (selected) target activations and APD topk activations. -> [i]."""
    assert target_post_acts.keys() == layer_acts.keys()
    total_act_dim = 0
    loss = 0.0
    for name in target_post_acts:
        total_act_dim += target_post_acts[name].shape[-1]
        loss = loss + ((target_post_acts[name] - layer_acts[name]) ** 2).sum(dim=-1)
    return (loss / total_act_dim).mean(dim=0)


def calc_lp_sparsity_loss(out: Tensor, attributions: Tensor, pnorm: float) -> Tensor:
    """Per-component Lp penalty on attributions (Lp mode, unused in topk configs)."""
    attributions = attributions / out.shape[-1]
    return (attributions.abs() + 1e-16) ** (pnorm * 0.5)


# ---------------------------------------------------------------------------
# LR schedules
# ---------------------------------------------------------------------------

def get_lr_schedule_fn(lr_schedule: str):
    if lr_schedule == "linear":
        return lambda step, steps: 1 - (step / steps)
    if lr_schedule == "constant":
        return lambda *_: 1.0
    if lr_schedule == "cosine":
        return lambda step, steps: 1.0 if steps == 1 else math.cos(0.5 * math.pi * step / (steps - 1))
    raise ValueError(lr_schedule)


def get_lr_with_warmup(step: int, steps: int, lr: float, lr_schedule_fn, lr_warmup_pct: float) -> float:
    warmup_steps = int(steps * lr_warmup_pct)
    if step < warmup_steps:
        return lr * (step / warmup_steps)
    return lr * lr_schedule_fn(step - warmup_steps, steps - warmup_steps)


# ---------------------------------------------------------------------------
# Optimization loop
# ---------------------------------------------------------------------------

@dataclass
class APDConfig:
    C: int
    topk: float | None
    batch_size: int
    steps: int
    lr: float
    seed: int = 0
    m: int | None = None
    batch_topk: bool = True
    lr_schedule: str = "constant"
    lr_warmup_pct: float = 0.0
    param_match_coeff: float | None = 1.0
    topk_recon_coeff: float | None = None
    act_recon_coeff: float | None = None
    schatten_coeff: float | None = None
    schatten_pnorm: float | None = None
    lp_sparsity_coeff: float | None = None
    pnorm: float | None = None
    unit_norm_matrices: bool = False
    attribution_type: str = "gradient"  # gradient | gim | ig | relp (attributions.py)
    selection: str = "batch_topk"       # batch_topk | greedy_add | greedy_prune (greedy.py)
    eps: float | None = None            # per-sample MSE threshold for the greedy selections
    print_freq: int = 1000
    extra: dict = field(default_factory=dict)


def remove_grad_parallel_to_subnetwork_vecs(A: Tensor, A_grad: Tensor) -> None:
    """Project out the gradient component parallel to each column of A (unit-norm mode)."""
    parallel_component = einops.einsum(A_grad, A, "... d_in m, ... d_in m -> ... m")
    A_grad -= einops.einsum(parallel_component, A, "... m, ... d_in m -> ... d_in m")


def optimize(
    model: nn.Module,          # APD model: forward(batch, topk_mask=None) -> (out, cache)
    target_model: nn.Module,   # target: forward(batch) -> (out, cache)
    config: APDConfig,
    generate_batch,            # callable(batch_size) -> [batch, n_instances, n_features]
    param_names: list[str],
    device: str,
    out_dir: Path | None = None,
    act_recon_transform=None,  # optional fn(post_acts dict) -> dict used for act_recon
                               # (resid-mlp: keep mlp_in layers only, apply ReLU)
    log_fn=None,
) -> dict:
    model.to(device)
    target_model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=0.0)
    lr_schedule_fn = get_lr_schedule_fn(config.lr_schedule)

    n_instances = next(iter(target_model.parameters())).shape[0]
    target_weights = {name: target_model.weights()[name] for name in param_names}
    n_params = sum(w.numel() for w in target_weights.values()) / n_instances

    metrics_log = []
    for step in range(config.steps + 1):
        if config.unit_norm_matrices:
            model.set_As_to_unit_norm()

        step_lr = get_lr_with_warmup(step, config.steps, config.lr, lr_schedule_fn, config.lr_warmup_pct)
        for group in opt.param_groups:
            group["lr"] = step_lr
        opt.zero_grad(set_to_none=True)

        batch = generate_batch(config.batch_size).to(device)
        target_out, target_cache = target_model(batch)
        out, apd_cache = model(batch)

        param_match_loss = None
        if config.param_match_coeff is not None:
            param_match_loss = calc_param_match_loss(
                target_weights, {n: model.weights()[n] for n in param_names}, n_params
            )

        from nano_apd.attributions import compute_attributions
        from nano_apd.greedy import greedy_add_mask, greedy_prune_mask

        post_weight_acts = {n: target_cache[n]["post"] for n in param_names}
        attributions = compute_attributions(
            config.attribution_type, target_model=target_model, apd_model=model,
            batch=batch, target_out=target_out, target_cache=target_cache,
            param_names=param_names, C=config.C,
        )

        lp_sparsity_loss_per_c = None
        if config.lp_sparsity_coeff is not None:
            lp_sparsity_loss_per_c = calc_lp_sparsity_loss(out, attributions, config.pnorm)

        topk_mask, topk_recon_loss, act_recon_loss, layer_acts_topk = None, None, None, None
        mask_l0 = None
        if config.topk is not None or config.selection != "batch_topk":
            if config.selection == "batch_topk":
                topk_mask = calc_topk_mask(attributions, config.topk, batch_topk=config.batch_topk)
            elif config.selection == "greedy_add":
                # iteration cap >> expected active components; binds only while the model
                # is still unfaithful early in training
                topk_mask = greedy_add_mask(model, batch, target_out, C=config.C,
                                            eps=config.eps, max_iter=min(32, config.C))
            elif config.selection == "greedy_prune":
                topk_mask = greedy_prune_mask(model, batch, target_out, attributions,
                                              eps=config.eps)
            elif config.selection == "greedy_prune_iter":
                from nano_apd.greedy import greedy_prune_iter_mask
                topk_mask = greedy_prune_iter_mask(model, batch, target_out, C=config.C,
                                                   eps=config.eps)
            else:
                raise ValueError(config.selection)
            mask_l0 = topk_mask.float().sum(dim=-1).mean().item()
            out_topk, topk_cache = model(batch, topk_mask=topk_mask)
            layer_acts_topk = {n: topk_cache[n]["post"] for n in param_names}
            if config.topk_recon_coeff is not None:
                topk_recon_loss = calc_recon_mse(out_topk, target_out)

        if config.act_recon_coeff is not None:
            assert layer_acts_topk is not None
            if act_recon_transform is not None:
                act_recon_loss = calc_act_recon(
                    act_recon_transform(post_weight_acts), act_recon_transform(layer_acts_topk)
                )
            else:
                act_recon_loss = calc_act_recon(post_weight_acts, layer_acts_topk)

        schatten_loss = None
        if config.schatten_coeff is not None:
            mask = topk_mask if topk_mask is not None else lp_sparsity_loss_per_c
            schatten_loss = calc_schatten_loss(
                As={n: model.As()[n] for n in param_names},
                Bs={n: model.Bs()[n] for n in param_names},
                mask=mask, p=config.schatten_pnorm, n_params=n_params,
            )

        lp_sparsity_loss = None
        if lp_sparsity_loss_per_c is not None:
            lp_sparsity_loss = lp_sparsity_loss_per_c.sum(dim=-1).mean(dim=0)

        out_recon_loss = calc_recon_mse(out, target_out)

        loss_terms = {
            "param_match_loss": (param_match_loss, config.param_match_coeff),
            "lp_sparsity_loss": (lp_sparsity_loss, config.lp_sparsity_coeff),
            "topk_recon_loss": (topk_recon_loss, config.topk_recon_coeff),
            "act_recon_loss": (act_recon_loss, config.act_recon_coeff),
            "schatten_loss": (schatten_loss, config.schatten_coeff),
        }
        loss = torch.tensor(0.0, device=device)
        for name, (term, coeff) in loss_terms.items():
            if coeff is not None:
                assert term is not None, f"{name} is None but coeff is set"
                loss = loss + coeff * term.mean()

        if step % config.print_freq == 0:
            record = {
                "step": step,
                "total_loss": loss.item(),
                "out_recon_loss": out_recon_loss.mean().item(),
                "lr": step_lr,
                **({"mask_l0": mask_l0} if mask_l0 is not None else {}),
                **{k: (v.mean().item() if v is not None else None) for k, (v, _) in loss_terms.items()},
            }
            metrics_log.append(record)
            msg = " ".join(f"{k}={v:.3e}" if isinstance(v, float) else f"{k}={v}"
                           for k, v in record.items() if v is not None)
            print(msg, flush=True)
            if log_fn is not None:
                log_fn(model, step)

        if step != config.steps:
            loss.backward()
            if config.unit_norm_matrices:
                model.fix_normalized_adam_gradients()
            opt.step()

    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), out_dir / "apd_model.pth")
        with open(out_dir / "apd_config.json", "w") as f:
            json.dump(asdict(config), f, indent=2)
        with open(out_dir / "metrics_log.json", "w") as f:
            json.dump(metrics_log, f, indent=2)
    return metrics_log[-1] if metrics_log else {}
