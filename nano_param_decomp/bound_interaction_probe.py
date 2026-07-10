"""Does causal INTERACTION reveal multiplicative binding where CO-ACTIVATION can't?

On BoundPairs (out_d = ReLU(x[2d]) * ReLU(x[2d+1])), each mechanism's two atoms -- layer-0 row d and
layer-1 row d -- are bound (drop either -> out_d dies) but read different features, so they DON'T
co-fire. We compare two atom-by-atom matrices:

  - co-activation: corr of atom activations across inputs. Bound partners read different features ->
    expected ~0 between (layer0 d, layer1 d). Co-activation can't see the binding.
  - interaction:   |d^2 L_recon / dm_a dm_b| at all-on, L = ||out(masked) - out(full)||^2. The product
    out_d = m0_d * m1_d * ... is BILINEAR in its two masks -> a large cross second-derivative for the
    bound pair, ~0 across pairs. Interaction should light up exactly the (layer0 d, layer1 d) blocks.
"""

import torch
import torch.nn.functional as F

from .toy_models import BoundPairs, feature_batch


def masked_out(model: BoundPairs, x: torch.Tensor, m0: torch.Tensor, m1: torch.Tensor) -> torch.Tensor:
    W0 = model.blocks[0]["proj"].weight  # [M, F]
    W1 = model.blocks[1]["proj"].weight
    z0 = x @ W0.t()  # [B, M]
    z1 = x @ W1.t()
    return F.relu(m0 * z0) * F.relu(m1 * z1)  # [B, M]


def interaction_matrix(model: BoundPairs, x: torch.Tensor) -> torch.Tensor:
    """|Hessian| of recon loss w.r.t. the 2M per-atom masks at all-on. Atom order: [layer0 rows, layer1 rows]."""
    M = model.n_mech
    m = torch.ones(2 * M, device=x.device, requires_grad=True)
    out_full = masked_out(model, x, torch.ones(M, device=x.device), torch.ones(M, device=x.device)).detach()
    out = masked_out(model, x, m[:M], m[M:])
    loss = F.mse_loss(out, out_full)
    grad = torch.autograd.grad(loss, m, create_graph=True)[0]  # [2M]
    H = torch.stack([torch.autograd.grad(grad[i], m, retain_graph=True)[0] for i in range(2 * M)])
    return H.abs().detach()


def coactivation_matrix(model: BoundPairs, x: torch.Tensor) -> torch.Tensor:
    W0 = model.blocks[0]["proj"].weight
    W1 = model.blocks[1]["proj"].weight
    a = torch.cat([F.relu(x @ W0.t()), F.relu(x @ W1.t())], dim=1)  # [B, 2M] atom activations
    a = a - a.mean(0, keepdim=True)
    c = (a.t() @ a) / (a.std(0).unsqueeze(1) * a.std(0).unsqueeze(0) * len(x) + 1e-8)
    return c.abs().detach()


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    M = 8
    model = BoundPairs(M).to(device)
    gen = torch.Generator(device=device).manual_seed(0)
    x = feature_batch(2 * M, 16384, 0.35, device, gen)  # denser so pairs co-occur enough to fire

    H = interaction_matrix(model, x)       # [2M, 2M], atoms [L0 0..M-1, L1 0..M-1]
    C = coactivation_matrix(model, x)

    # for each mechanism d, the bound partners are atoms d (L0) and M+d (L1).
    def partner_vs_other(mat: torch.Tensor) -> tuple[float, float]:
        partner, other = [], []
        for d in range(M):
            for e in range(M):
                v = mat[d, M + e].item()  # L0 d  vs  L1 e
                (partner if e == d else other).append(v)
        return sum(partner) / len(partner), sum(other) / len(other)

    hp, ho = partner_vs_other(H)
    cp, co = partner_vs_other(C)
    print("=== bound-pair signal: (layer0 d) vs (layer1 e) ===", flush=True)
    print(f"INTERACTION  partner(e=d)={hp:.4f}   other(e!=d)={ho:.4f}   ratio={hp/(ho+1e-9):.1f}x", flush=True)
    print(f"CO-ACTIVATION partner(e=d)={cp:.4f}   other(e!=d)={co:.4f}   ratio={cp/(co+1e-9):.1f}x", flush=True)
    print(f"\nverdict: interaction {'SEPARATES' if hp > 5 * ho else 'does NOT separate'} bound pairs; "
          f"co-activation {'separates' if cp > 5 * co else 'does NOT separate'} them", flush=True)
    print("BOUND_PROBE_DONE", flush=True)


if __name__ == "__main__":
    main()
