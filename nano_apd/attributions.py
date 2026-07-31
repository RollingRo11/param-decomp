"""Attribution methods for APD, pluggable in place of the original gradient attribution.

All return [batch, n_instances, C] scores. Terminology follows the source papers:

- "gradient": original APD (arXiv 2501.14926) — for each output dim o, sum over layers of
  (dy_o/dh_layer . component_act_c,layer), squared, summed over o. Gradients through the
  target model graph; component acts = detached target pre-acts @ component weights.
- "ig": integrated gradients over the component-mask path — gradients of the APD model
  output w.r.t. the component mask m, averaged along the straight path m = alpha*1,
  alpha in (0,1). Captures interaction effects that the single-point gradient misses
  (the completeness sum over c approximates y(1) - y(0) per output dim).
- "relp": Relevance Patching (arXiv 2508.21258) adapted to APD — replace dy_o/dh_layer
  with LRP-epsilon propagation coefficients rho computed on the target model, then score
  components as sum_o (sum_layers rho_layer,o . component_act_c,layer)^2. LRP rules used:
  0-rule at the readout, epsilon-rule through linear layers and the residual split,
  identity through ReLU with the bias absorbing its epsilon-rule share.
- "gim" (arXiv 2505.17630): GIM's modifications (temperature-adjusted softmax gradients,
  layernorm freeze, gradient normalization at multiplicative interactions) target
  transformer components that do not exist in the resid-MLP toys, where GIM's backward
  equals the vanilla gradient. Registered as an alias of "gradient" so configs are
  explicit; it only becomes distinct on architectures with those components.
"""

import einops
import torch
from torch import Tensor

from nano_apd.apd import calc_grad_attributions


def calc_component_acts(pre_weight_acts: dict[str, Tensor],
                        component_weights: dict[str, Tensor]) -> dict[str, Tensor]:
    return {
        name: einops.einsum(
            pre_weight_acts[name].detach().clone(), component_weights[name],
            "batch i d_in, i C d_in d_out -> batch i C d_out",
        )
        for name in component_weights
    }


# ---------------------------------------------------------------------------
# Integrated gradients over the component-mask path
# ---------------------------------------------------------------------------

def calc_ig_mask_attributions(apd_model, batch: Tensor, C: int, n_alpha: int = 8) -> Tensor:
    """IG along m = alpha*1 for the APD model's own output; squared per output dim and
    summed, mirroring APD's squared-attribution convention."""
    device = batch.device
    b, i = batch.shape[0], batch.shape[1]
    total = None
    for s in range(n_alpha):
        alpha = (s + 0.5) / n_alpha
        mask = torch.full((b, i, C), alpha, device=device, requires_grad=True)
        out, _ = apd_model(batch, topk_mask=mask)
        out_dim = out.shape[-1]
        eye = torch.eye(out_dim, device=device, dtype=out.dtype)
        summed_out = einops.einsum(out, "batch i d_out -> d_out")
        grads = torch.autograd.grad(summed_out, mask, grad_outputs=eye,
                                    is_grads_batched=True)[0]  # [o, b, i, C]
        total = grads if total is None else total + grads
    ig = total / n_alpha                                       # [o, b, i, C]
    return einops.einsum(ig**2, "o batch i C -> batch i C")


# ---------------------------------------------------------------------------
# RelP (LRP-epsilon propagation coefficients) for the resid-MLP target
# ---------------------------------------------------------------------------

def _stab(x: Tensor, eps: float) -> Tensor:
    return x + eps * torch.where(x >= 0, 1.0, -1.0)


def calc_relp_coefficients(target_model, target_cache: dict, eps: float = 1e-6
                           ) -> dict[str, Tensor]:
    """LRP-epsilon propagation coefficients rho at each post-weight activation of a
    ResidMLPModel, vectorized over output dims: rho[name] has shape [o, batch, i, d_out]
    and plays the role of dy_o/dh_name in the attribution formula.

    Derivation (per output o, epsilon-rule everywhere, identity through ReLU):
      readout    R_final = W_U[:, o] * resid_final          (0-rule; sums to y_o)
      resid add  rho_resid = R_resid / stab(resid_out): shared by the h_in and m_out paths
      mlp_out    R_a = a * (W_out @ rho_mout)
      relu+bias  R_z = R_a * z / stab(z + bias)             (bias absorbs its share)
      mlp_in     R_hin += h_in * (W_in @ rho_z)
    """
    cfg = target_model.config
    n_layers = cfg.n_layers
    W_U = target_model.W_U  # [i, e, f]

    pre_in = [target_cache[f"layers.{l}.mlp_in"]["pre"].detach() for l in range(n_layers)]
    z = [target_cache[f"layers.{l}.mlp_in"]["post"].detach() for l in range(n_layers)]
    m_out = [target_cache[f"layers.{l}.mlp_out"]["post"].detach() for l in range(n_layers)]
    resid_final = pre_in[-1] + m_out[-1]                        # [b, i, e]

    # relevance at resid_final, per output o: [o, b, i, e]
    R = einops.einsum(W_U.detach(), resid_final, "i e f, b i e -> f b i e")

    rho = {}
    for l in range(n_layers - 1, -1, -1):
        resid_out = pre_in[l] + m_out[l]
        rho_resid = R / _stab(resid_out, eps)                   # [o, b, i, e]
        rho[f"layers.{l}.mlp_out"] = rho_resid                  # coefficient at m_out
        R_hin = rho_resid * pre_in[l]

        # through mlp_out: relevance at hidden a, then coefficient at z
        W_out = target_model.mlp_out[l].weight.detach()         # [i, m, e]
        bias = (target_model.bias1[l].detach() if target_model.bias1 is not None
                else torch.zeros_like(z[l][0]))
        a = torch.relu(z[l] + bias)
        back = einops.einsum(W_out, rho_resid, "i m e, o b i e -> o b i m")
        R_a = a * back
        # identity through ReLU; epsilon-rule split between z and the bias
        rho[f"layers.{l}.mlp_in"] = R_a / _stab(z[l] + bias, eps)  # coefficient at z

        W_in = target_model.mlp_in[l].weight.detach()           # [i, e, m]
        back_in = einops.einsum(W_in, rho[f"layers.{l}.mlp_in"], "i e m, o b i m -> o b i e")
        R = R_hin + pre_in[l] * back_in

    return rho


def calc_relp_attributions(target_model, target_cache: dict,
                           component_weights: dict[str, Tensor], C: int,
                           eps: float = 1e-6) -> Tensor:
    rho = calc_relp_coefficients(target_model, target_cache, eps=eps)
    pre_weight_acts = {n: target_cache[n]["pre"] for n in component_weights}
    component_acts = calc_component_acts(pre_weight_acts, component_weights)
    first = next(iter(rho.values()))
    out_dim = first.shape[0]
    feature_attr = None
    for name in component_weights:
        term = einops.einsum(rho[name], component_acts[name],
                             "o batch i d_out, batch i C d_out -> o batch i C")
        feature_attr = term if feature_attr is None else feature_attr + term
    assert feature_attr.shape[0] == out_dim
    return einops.einsum(feature_attr**2, "o batch i C -> batch i C")


# ---------------------------------------------------------------------------
# IFR (Information Flow Routes, arXiv 2403.00824) — ALTI proximity contributions
# ---------------------------------------------------------------------------

def _proximity_share(parts: Tensor, total: Tensor) -> Tensor:
    """ALTI proximity rule: importance(z_j, y) = max(||y||_1 - ||z_j - y||_1, 0),
    normalized over the incoming edges. parts [b, i, C, d], total [b, i, d] -> [b, i, C]."""
    prox = torch.clamp(
        total.abs().sum(-1, keepdim=False).unsqueeze(-1)
        - (parts - total.unsqueeze(-2)).abs().sum(-1),
        min=0.0,
    )
    return prox / (prox.sum(dim=-1, keepdim=True) + 1e-12)


def calc_ifr_attributions(target_model, apd_model, batch: Tensor, target_cache: dict,
                          param_names: list[str], C: int) -> Tensor:
    """Gradient-free contribution attribution in the style of Information Flow Routes:
    a component's score is its per-node proximity share (at the layer outputs it feeds),
    weighted by that layer-node's own proximity share at the readout node. One forward
    pass, no gradients; shares are normalized per node by construction."""
    cfg = target_model.config
    n_layers = cfg.n_layers
    pre_weight_acts = {n: target_cache[n]["pre"] for n in param_names}
    component_acts = calc_component_acts(
        pre_weight_acts, {n: apd_model.component_weights()[n] for n in param_names})

    # downstream share of each layer's m_out at the output node (edges: W_U @ m_out_l
    # for each layer, plus the W_E passthrough)
    W_U = target_model.W_U.detach()
    m_out = [target_cache[f"layers.{l}.mlp_out"]["post"].detach() for l in range(n_layers)]
    resid_final = target_cache[f"layers.{n_layers-1}.mlp_in"]["pre"].detach() + m_out[-1]
    y = einops.einsum(resid_final, W_U, "b i e, i e f -> b i f")
    edges = [einops.einsum(m, W_U, "b i e, i e f -> b i f") for m in m_out]
    passthrough = y - sum(edges)
    edge_stack = torch.stack(edges + [passthrough], dim=-2)      # [b, i, n_layers+1, f]
    layer_share = _proximity_share(edge_stack, y)[..., :n_layers]  # [b, i, n_layers]

    attr = None
    for l in range(n_layers):
        w_l = layer_share[..., l].unsqueeze(-1)                  # [b, i, 1]
        for proj in ("mlp_in", "mlp_out"):
            name = f"layers.{l}.{proj}"
            node_total = target_cache[name]["post"].detach()
            share = _proximity_share(component_acts[name], node_total)  # [b, i, C]
            term = w_l * share
            attr = term if attr is None else attr + term
    return attr


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

def compute_attributions(attribution_type: str, *, target_model, apd_model, batch: Tensor,
                         target_out: Tensor, target_cache: dict, param_names: list[str],
                         C: int) -> Tensor:
    if attribution_type in ("gradient", "gim"):
        return calc_grad_attributions(
            target_out=target_out,
            pre_weight_acts={n: target_cache[n]["pre"] for n in param_names},
            post_weight_acts={n: target_cache[n]["post"] for n in param_names},
            component_weights={n: apd_model.component_weights()[n] for n in param_names},
            C=C,
        )
    if attribution_type == "ig":
        return calc_ig_mask_attributions(apd_model, batch, C)
    if attribution_type == "relp":
        return calc_relp_attributions(
            target_model, target_cache,
            {n: apd_model.component_weights()[n] for n in param_names}, C,
        )
    if attribution_type == "ifr":
        return calc_ifr_attributions(target_model, apd_model, batch, target_cache,
                                     param_names, C)
    raise ValueError(f"unknown attribution_type: {attribution_type}")
