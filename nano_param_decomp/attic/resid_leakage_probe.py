"""Verify the cached 2-layer residual MLP genuinely realizes ~100 separable per-feature mechanisms.

The target is element-wise (y_i = x_i + ReLU(x_i)), so the *task* is 100 independent scalar maps.
But the trained net packs 100 features into 40 neurons (superposition), so features share atoms and
the ReLU can couple them. This probe checks whether the *trained* net keeps them separable:

  - one-hot probe: drive feature i alone (value 1.0) -> output should be ~2 on channel i, ~0 elsewhere
  - leakage matrix L[i, j] = output_j when only feature i is on; near-diagonal => separable
  - pairwise interference: drive {i, j} together, compare to sum of singles => measures ReLU coupling
"""

import torch
import torch.nn.functional as F

from .toy_models import ResidMLP, target_fn


def load_cached(path: str, device: torch.device) -> ResidMLP:
    sd = torch.load(path, map_location="cpu")
    nf, d_embed = sd["W_E"].shape
    d_mlp_layer = sd["blocks.0.in_proj.weight"].shape[0]
    n_layers = sum(1 for k in sd if k.endswith("in_proj.weight"))
    model = ResidMLP(nf, d_embed, d_mlp_layer * n_layers, n_layers)
    model.load_state_dict(sd)
    return model.to(device).eval()


@torch.no_grad()
def leakage_matrix(model: ResidMLP, value: float, device: torch.device) -> torch.Tensor:
    """L[i, j] = output on channel j when ONLY feature i is active at `value`. Shape [nf, nf]."""
    nf = model.n_features
    x = torch.eye(nf, device=device) * value  # row i = one-hot feature i
    return model(x)  # [nf, nf]


@torch.no_grad()
def report(path: str) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_cached(path, device)
    nf = model.n_features
    value = 1.0
    tgt_diag = value + max(value, 0.0)  # x + relu(x) at x=value => 2.0

    L = leakage_matrix(model, value, device)
    diag = L.diag()
    off = L - torch.diag(diag)

    diag_err = (diag - tgt_diag).abs()
    off_abs = off.abs()
    # per-feature: is channel i right AND nothing else leaks much?
    leak_per_feat = off_abs.max(dim=1).values  # worst leak caused by feature i
    clean = ((diag_err < 0.1 * tgt_diag) & (leak_per_feat < 0.1 * tgt_diag)).sum().item()

    print(f"=== one-hot separability ({nf} features, target diag={tgt_diag:.1f}) ===", flush=True)
    print(f"diagonal:   mean={diag.mean():.3f}  min={diag.min():.3f}  max={diag.max():.3f}", flush=True)
    print(f"diag error: mean={diag_err.mean():.4f}  max={diag_err.max():.4f}", flush=True)
    print(f"off-diag |leak|: mean={off_abs.mean():.4f}  max={off_abs.max():.4f}", flush=True)
    print(f"signal/leak ratio: {diag.mean().item() / (off_abs.mean().item() + 1e-9):.1f}x", flush=True)
    print(f"clean features (diag ok AND leak <10%): {clean}/{nf}", flush=True)

    # pairwise interference: does driving two features together = sum of singles? (ReLU coupling)
    gen = torch.Generator(device=device).manual_seed(0)
    pairs = torch.randint(0, nf, (2, 2000), generator=gen, device=device)
    pairs = pairs[:, pairs[0] != pairs[1]]
    i, j = pairs[0], pairs[1]
    xi = F.one_hot(i, nf).float() * value
    xj = F.one_hot(j, nf).float() * value
    both = model(xi + xj)
    singles = model(xi) + model(xj) - model(torch.zeros_like(xi))  # subtract bias baseline
    interference = (both - singles).abs().mean().item()
    print(f"\npairwise interference (|both - sum-of-singles|): {interference:.4f}", flush=True)

    # sanity: overall recon on the real sparse distribution
    x = (torch.rand(8192, nf, device=device, generator=gen) < 0.01).float() \
        * torch.rand(8192, nf, device=device, generator=gen)
    mse = F.mse_loss(model(x), target_fn(x)).item()
    print(f"recon MSE on sparse eval: {mse:.5f}", flush=True)
    print("PROBE_DONE", flush=True)


if __name__ == "__main__":
    report("/tmp/toy/resid_2l.pt")
