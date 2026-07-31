"""Per-sample component selection by an epsilon-faithfulness criterion instead of top-k.

Two procedures (no k anywhere; per-sample MSE against the target output is the only
stopping condition):

- greedy_add (matching-pursuit-like, stopping on faithfulness): start with no components
  active; attribute the remaining error to components via the gradient of the per-sample
  error w.r.t. the component mask (= alignment between the residual error and each
  component's marginal contribution); activate the best proposal per sample; stop a
  sample once its reconstruction MSE < eps.

- greedy_prune: start with all components active; the chosen attribution method ranks
  components (ascending = most removable); per sample, binary-search the longest
  removable prefix of that ranking such that MSE stays < eps, then certify by a real
  forward pass (attribution proposes, reconstruction certifies). Falls back to smaller
  removals if certification fails.

Both return a float {0,1} mask [batch, n_instances, C] usable exactly like the top-k mask.
"""

import einops
import torch
from torch import Tensor


def _sample_mse(out: Tensor, target_out: Tensor) -> Tensor:
    """[b, i, d_out] -> [b, i] per-sample MSE over output dims."""
    return ((out - target_out) ** 2).mean(dim=-1)


@torch.enable_grad()
def greedy_add_mask(apd_model, batch: Tensor, target_out: Tensor, C: int, eps: float,
                    max_iter: int | None = None) -> Tensor:
    b, i = batch.shape[0], batch.shape[1]
    device = batch.device
    target_out = target_out.detach()
    mask = torch.zeros(b, i, C, device=device)
    max_iter = C if max_iter is None else max_iter

    for _ in range(max_iter):
        mask_var = mask.clone().requires_grad_(True)
        out, _ = apd_model(batch, topk_mask=mask_var)
        err = _sample_mse(out, target_out)                     # [b, i]
        not_done = err >= eps
        if not not_done.any():
            break
        grad = torch.autograd.grad(err.sum(), mask_var)[0]     # [b, i, C]
        # most negative gradient = activating this component reduces the error fastest;
        # exclude already-active components
        score = torch.where(mask.bool(), torch.inf, grad)
        proposal = score.argmin(dim=-1)                        # [b, i]
        update = torch.zeros_like(mask)
        update.scatter_(-1, proposal.unsqueeze(-1), 1.0)
        mask = torch.clamp(mask + update * not_done.unsqueeze(-1).float(), max=1.0)
    return mask.detach()


def _mask_from_prefix_len(order: Tensor, k: Tensor, C: int) -> Tensor:
    """order: [b, i, C] ascending-removability component indices; k: [b, i] number of
    leading entries of `order` to REMOVE. Returns float mask [b, i, C]."""
    ranks = torch.empty_like(order)
    ranks.scatter_(-1, order, torch.arange(C, device=order.device).expand_as(order))
    return (ranks >= k.unsqueeze(-1)).float()


@torch.no_grad()
def greedy_prune_mask(apd_model, batch: Tensor, target_out: Tensor,
                      attributions: Tensor, eps: float) -> Tensor:
    b, i = batch.shape[0], batch.shape[1]
    C = attributions.shape[-1]
    target_out = target_out.detach()
    order = attributions.argsort(dim=-1)                       # ascending = remove first

    lo = torch.zeros(b, i, dtype=torch.long, device=batch.device)
    hi = torch.full((b, i), C, dtype=torch.long, device=batch.device)
    steps = max(1, int(torch.tensor(float(C)).log2().ceil().item()))
    for _ in range(steps + 1):
        if (lo >= hi).all():
            break
        mid = (lo + hi + 1) // 2
        out, _ = apd_model(batch, topk_mask=_mask_from_prefix_len(order, mid, C))
        ok = _sample_mse(out, target_out) < eps
        lo = torch.where(ok, mid, lo)
        hi = torch.where(ok, hi, mid - 1)

    # certify (the error curve need not be monotone in the prefix length); on failure
    # retry with half the removal, then fall back to removing nothing for that sample
    for fallback in ("half", "none"):
        mask = _mask_from_prefix_len(order, lo, C)
        out, _ = apd_model(batch, topk_mask=mask)
        bad = (_sample_mse(out, target_out) >= eps) & (lo > 0)
        if not bad.any():
            break
        lo = torch.where(bad, lo // 2 if fallback == "half" else torch.zeros_like(lo), lo)
    return _mask_from_prefix_len(order, lo, C)


@torch.enable_grad()
def greedy_prune_iter_mask(apd_model, batch: Tensor, target_out: Tensor, C: int,
                           eps: float, max_rounds: int = 24) -> Tensor:
    """Recompute-per-round pruning (the strict reading of "attribution proposes,
    reconstruction certifies" with re-attribution after each removal).

    Note: APD's gradient attribution is mask-independent, so re-ranking with it each
    round would be a no-op. Proposals here instead come from the current partial model:
    d(per-sample recon error)/d(mask), i.e. how much each still-active component's
    removal is predicted to change the error. Each round tentatively removes the r
    least-harmful active components per sample (r adapts: kept on success, halved on
    certification failure), and a real forward pass certifies err < eps per sample.
    Stops when r reaches 0 everywhere or max_rounds."""
    b, i = batch.shape[0], batch.shape[1]
    device = batch.device
    target_out = target_out.detach()
    mask = torch.ones(b, i, C, device=device)
    r = torch.full((b, i), C // 2, dtype=torch.long, device=device)

    for _ in range(max_rounds):
        if (r == 0).all():
            break
        mask_var = mask.clone().requires_grad_(True)
        out, _ = apd_model(batch, topk_mask=mask_var)
        err = _sample_mse(out, target_out)
        grad = torch.autograd.grad(err.sum(), mask_var)[0]       # [b, i, C]
        # removal effect of an ON component ~ -grad (first order); rank ON components
        # by predicted harm of removal, ascending (least harmful first)
        harm = torch.where(mask.bool(), -grad, torch.inf)
        order = harm.argsort(dim=-1)                             # off components sort last
        ranks = torch.empty_like(order)
        ranks.scatter_(-1, order, torch.arange(C, device=device).expand_as(order))
        proposal = mask * (ranks >= r.unsqueeze(-1)).float()     # remove r least harmful

        with torch.no_grad():
            out_p, _ = apd_model(batch, topk_mask=proposal)
            ok = (_sample_mse(out_p, target_out) < eps) & (r > 0)
        mask = torch.where(ok.unsqueeze(-1), proposal, mask)
        n_active = mask.sum(dim=-1).long()
        r = torch.where(ok, torch.minimum(r, n_active), r // 2)
        r = torch.minimum(r, n_active)
    return mask.detach()


def mean_l0(mask: Tensor) -> float:
    """Mean number of active components per sample (for logging)."""
    return einops.reduce(mask, "b i C -> b i", "sum").mean().item()
