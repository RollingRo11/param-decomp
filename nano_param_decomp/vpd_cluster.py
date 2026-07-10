"""VPD's post-hoc MDL clustering of rank-1 subcomponents into cross-layer parameter components
(VPD paper Appendix A.8), and a head-to-head vs our end-to-end APD-basis on the cross-layer ResidMLP.

The question (Rohan): VPD produces rank-1 atoms confined to single matrices; the paper's *explicit*
clustering (App A.8) is how it assembles them into whole-network components after the fact. Does that
2-stage pipeline (train VPD -> cluster) recover the cross-layer, per-feature mechanisms as well as
apd_mask's end-to-end whole-network basis?

Method (A.8), reproduced here:
  - Binarize atom causal importances: g_i = 1 if CI_i > tau (tau=0.01).  A cluster's importance is the
    OR of its atoms'.
  - Coactivation s_{i,j} = sum_{datapoints} g_i g_j ; diagonal s_i = firing count.
  - MDL cost L = sum_i s_i (log2 k + alpha * rank_i), rank of a cluster = #atoms in it (all rank-1).
  - Stochastic hierarchical merging: start each atom alone; each step compute merge cost DeltaL for
    every pair (dictionary-reduction + index-encoding + rank-penalty terms), rank pairs ascending,
    sample a rank via the inverse-CDF exp(-gamma*J) (gamma=0.2), merge, until one cluster remains.
    Take the partition at minimum total MDL along the trajectory (where DeltaL crossed 0).
  - alpha trades #components vs per-component complexity; we sweep it.

We then score the resulting clusters with the SAME recovery probe as apd_mask.feature_recovery_resid
(injectivity / purity / cross_layer / keep_only), so the comparison is apples-to-apples.

Run:  CUDA_VISIBLE_DEVICES=0 python -m nano_param_decomp.vpd_cluster
"""

import math
import os

import torch
import torch.nn.functional as F
from torch import Tensor

from .run import ComponentLinear, clear_wrapper_masks, set_wrapper_masks
from .toy_decompose import ToyConfig, decompose_toy


# --- MDL clustering (App A.8) ----------------------------------------------------------------------


def _one_merge_run(g_bin: Tensor, ranks: Tensor, alpha: float, gamma: float,
                   gen: torch.Generator | None) -> tuple[float, list[list[int]]]:
    """One stochastic hierarchical merge to a single cluster; return (min MDL seen, partition there)."""
    device = g_bin.device
    n = g_bin.shape[1]
    G = (g_bin.t() > 0.5).float().clone()        # [k, N] group activation = OR of members
    r = ranks.clone().float()                    # [k] group rank = #atoms
    members: list[list[int]] = [[i] for i in range(n)]

    def total_mdl() -> float:
        k = G.shape[0]
        s = G.sum(dim=1)
        return float((s * (math.log2(k) + alpha * r)).sum())

    best_L = total_mdl()
    best_members = [list(m) for m in members]

    while G.shape[0] > 1:
        k = G.shape[0]
        S = G @ G.t()                            # [k,k] coactivation; diag = firing count
        s = torch.diagonal(S)
        s_tot = s.sum()
        s_i, s_j = s.view(k, 1), s.view(1, k)
        s_union = s_i + s_j - S                  # |g_i OR g_j|
        r_i, r_j = r.view(k, 1), r.view(1, k)
        log_k = math.log2(k)
        log_km1 = math.log2(k - 1) if k > 1 else 0.0
        dict_red = (s_tot - s_i - s_j) * (log_km1 - log_k)
        index = s_union * log_km1 - s_i * log_k - s_j * log_k
        rank_pen = alpha * (s_union * (r_i + r_j) - s_i * r_i - s_j * r_j)
        dL = dict_red + index + rank_pen

        iu = torch.triu_indices(k, k, offset=1, device=device)
        pair_dL = dL[iu[0], iu[1]]
        order = torch.argsort(pair_dL)           # ascending: rank 0 = lowest cost
        P = pair_dL.numel()
        u = torch.rand((), device=device, generator=gen).item()
        if gamma > 0:
            J = int(-math.log(1 - u * (1 - math.exp(-gamma * P))) / gamma)
        else:
            J = int(u * P)
        J = max(0, min(P - 1, J))
        pick = order[J].item()
        i, j = int(iu[0][pick]), int(iu[1][pick])
        if i > j:
            i, j = j, i

        G[i] = torch.maximum(G[i], G[j])
        r[i] = r[i] + r[j]
        members[i] = members[i] + members[j]
        keep = [t for t in range(k) if t != j]
        G, r, members = G[keep], r[keep], [members[t] for t in keep]

        L = total_mdl()
        if L < best_L:
            best_L = L
            best_members = [list(m) for m in members]

    return best_L, best_members


def mdl_cluster(g_bin: Tensor, ranks: Tensor, alpha: float, gamma: float = 0.2,
                gen: torch.Generator | None = None, restarts: int = 12) -> list[list[int]]:
    """Stochastic hierarchical MDL merge (App A.8), `restarts` times; return the global min-MDL
    partition. A single stochastic path can wander past the best partition, so exploration across
    restarts (the point of the stochasticity) is needed to reliably find it. `g_bin` [N, n] in {0,1}."""
    best_L, best = float("inf"), [[i] for i in range(g_bin.shape[1])]
    for _ in range(restarts):
        L, m = _one_merge_run(g_bin, ranks, alpha, gamma, gen)
        if L < best_L:
            best_L, best = L, m
    return best


# --- atom layout + binary CI -----------------------------------------------------------------------


def atom_layout(cfg: ToyConfig, order: list[str]) -> tuple[list[str], list[int]]:
    """Global atom index -> (module_name, layer). Returns (module_per_atom, layer_per_atom)."""
    module_per_atom: list[str] = []
    layer_per_atom: list[int] = []
    for m in order:
        layer = 0 if "blocks.0" in m else 1
        for _ in range(cfg.C_per_module[m]):
            module_per_atom.append(m)
            layer_per_atom.append(layer)
    return module_per_atom, layer_per_atom


@torch.no_grad()
def binary_ci(model, wrappers: dict[str, ComponentLinear], ci, x: Tensor,
              order: list[str], tau: float) -> Tensor:
    """[B, n_atoms] binarized causal importances (concatenated in `order`)."""
    clear_wrapper_masks(wrappers)
    _ = model(x)
    acts = {n: w.last_input for n, w in wrappers.items()}
    ci_lower, _u, _r = ci(acts)  # [B, n_atoms] concatenated
    return (ci_lower > tau).float()


# --- cluster recovery probe (mirrors apd_mask.feature_recovery_resid) ------------------------------


@torch.no_grad()
def cluster_recovery(model, wrappers: dict[str, ComponentLinear], ci, clusters: list[list[int]],
                     cfg: ToyConfig, order: list[str], nf: int, tau: float, device: torch.device,
                     n_probe: int = 128) -> dict[str, float]:
    module_per_atom, layer_per_atom = atom_layout(cfg, order)
    K = len(clusters)
    # per-cluster (module -> local atom indices) for masked forwards, and layers spanned
    cluster_local: list[dict[str, list[int]]] = []
    cluster_layers: list[set[int]] = []
    base = {m: 0 for m in order}
    offset, off_by_module = 0, {}
    for m in order:
        off_by_module[m] = offset
        offset += cfg.C_per_module[m]
    for cl in clusters:
        loc: dict[str, list[int]] = {m: [] for m in order}
        layers: set[int] = set()
        for g in cl:
            m = module_per_atom[g]
            loc[m].append(g - off_by_module[m])
            layers.add(layer_per_atom[g])
        cluster_local.append(loc)
        cluster_layers.append(layers)

    # pass 1: per-feature cluster activation (OR of member atoms), mean over probe -> assignment
    A = torch.zeros(nf, K, device=device)
    for f in range(nf):
        x = torch.zeros(n_probe, nf, device=device)
        x[:, f] = torch.rand(n_probe, device=device)
        gb = binary_ci(model, wrappers, ci, x, order, tau)  # [n_probe, n_atoms]
        for c, cl in enumerate(clusters):
            if cl:
                A[f, c] = gb[:, cl].amax(dim=1).mean()  # OR over cluster atoms, then mean over probe
    assigned = A.argmax(dim=1)
    top1 = A.gather(1, assigned[:, None]).squeeze(1)
    active = top1 > 0.5
    if not bool(active.any()):
        return {"n_clusters": K, "active_frac": 0.0, "injectivity": 0.0, "purity": 0.0,
                "cross_layer": 0.0, "keep_only": float("nan"), "keep_only_off": float("nan")}
    purity = (top1 / (A.sum(dim=1) + 1e-8))[active].mean().item()
    injectivity = assigned[active].unique().numel() / int(active.sum())
    cross_layer = sum(len(cluster_layers[c]) > 1 for c in assigned[active].tolist()) / int(active.sum())

    # causal sufficiency: keep ONLY the assigned cluster's atoms on, recon feature f's own output dim
    keep_only, keep_off = [], []
    zeros_delta = {m: torch.zeros(n_probe, device=device) for m in order}
    for f in range(nf):
        if not bool(active[f]):
            continue
        x = torch.zeros(n_probe, nf, device=device)
        x[:, f] = torch.rand(n_probe, device=device)
        clear_wrapper_masks(wrappers)
        tgt = model(x).detach()
        var_f = tgt[:, f].var() + 1e-8
        c = int(assigned[f])
        masks = {}
        for m in order:
            mk = torch.zeros(n_probe, cfg.C_per_module[m], device=device)
            if cluster_local[c][m]:
                mk[:, cluster_local[c][m]] = 1.0
            masks[m] = mk
        set_wrapper_masks(wrappers, masks, zeros_delta, routing=None)
        pred = model(x)
        clear_wrapper_masks(wrappers)
        keep_only.append((F.mse_loss(pred[:, f], tgt[:, f]) / var_f).item())
        off_masks = {m: torch.zeros(n_probe, cfg.C_per_module[m], device=device) for m in order}
        set_wrapper_masks(wrappers, off_masks, zeros_delta, routing=None)
        off = model(x)
        clear_wrapper_masks(wrappers)
        keep_off.append((F.mse_loss(off[:, f], tgt[:, f]) / var_f).item())
    return {
        "n_clusters": K,
        "active_frac": int(active.sum()) / nf,
        "injectivity": injectivity,
        "purity": purity,
        "cross_layer": cross_layer,
        "keep_only": sum(keep_only) / len(keep_only),
        "keep_only_off": sum(keep_off) / len(keep_off),
    }


# --- main: run VPD, cluster (sweep alpha), score, compare ------------------------------------------


def main() -> None:
    import copy

    from .toy_models import ResidMLP, feature_batch

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    nf, d_embed, d_mlp, n_layers, fprob = 100, 256, 40, 2, 0.01
    steps = int(os.environ.get("STEPS", "8000"))
    tau = float(os.environ.get("TAU", "0.01"))
    seed = int(os.environ.get("SEED", "0"))
    base = ResidMLP(nf, d_embed, d_mlp, n_layers, seed=0).to(device)
    base.load_state_dict(torch.load("/tmp/toy/resid_2l_apd_dmlp40.pt", weights_only=True))
    gen = torch.Generator(device=device).manual_seed(123)

    def data_fn() -> Tensor:
        return feature_batch(nf, 2048, fprob, device, gen)

    dml = d_mlp // n_layers
    cpm = {f"blocks.{i}.{p}": dml * 2 for i in range(n_layers) for p in ("in_proj", "out_proj")}
    order = sorted(cpm)
    cfg = ToyConfig(C_per_module=cpm, mode="vpd", n_components=200, n_steps=steps, warmup_steps=500,
                    batch=2048, lr=3e-3, coeff_imp=3e-2, tau_start=1.0, tau_end=1.0)

    model = copy.deepcopy(base)
    print(f"training VPD (resid, {steps} steps, {sum(cpm.values())} atoms) ...", flush=True)
    out = decompose_toy(model, data_fn, cfg, device)
    wrappers, ci = out["wrappers"], out["ci"]
    print(f"VPD done: recon_ci={out['recon_ci']:.4f} L0(atoms)={out['l0']:.2f}", flush=True)

    # coactivation batch (bigger N for stable stats)
    gb = torch.cat([binary_ci(model, wrappers, ci, data_fn(), order, tau) for _ in range(8)], dim=0)
    n_atoms = gb.shape[1]
    ranks = torch.ones(n_atoms, device=device)
    firing = gb.mean(dim=0)
    print(f"coactivation batch N={gb.shape[0]}, atoms firing (>tau) mean density={firing.mean():.4f}, "
          f"n atoms ever firing={(firing>0).sum().item()}/{n_atoms}", flush=True)

    cgen = torch.Generator(device=device).manual_seed(seed)
    print(f"\n{'alpha':>8} {'#clust':>7} {'active':>7} {'inject':>7} {'purity':>7} {'xlayer':>7} {'keep':>7} {'(off)':>7}")
    if os.environ.get("SINGLETON", "0") == "1":  # no clustering at all: every atom its own unit
        singles = [[i] for i in range(n_atoms)]
        rec = cluster_recovery(model, wrappers, ci, singles, cfg, order, nf, tau, device)
        print(f"{'raw':>8} {rec['n_clusters']:>7d} {rec['active_frac']:>7.2f} {rec['injectivity']:>7.2f} "
              f"{rec['purity']:>7.2f} {rec['cross_layer']:>7.2f} {rec['keep_only']:>7.3f} {rec['keep_only_off']:>7.3f}",
              flush=True)
    alphas_env = os.environ.get("ALPHAS", "0.01,0.1,1,10")
    for alpha in ([float(a) for a in alphas_env.split(",")] if alphas_env else []):
        clusters = mdl_cluster(gb, ranks, alpha, gamma=0.2, gen=cgen)
        rec = cluster_recovery(model, wrappers, ci, clusters, cfg, order, nf, tau, device)
        print(f"{alpha:>8g} {rec['n_clusters']:>7d} {rec['active_frac']:>7.2f} {rec['injectivity']:>7.2f} "
              f"{rec['purity']:>7.2f} {rec['cross_layer']:>7.2f} {rec['keep_only']:>7.3f} {rec['keep_only_off']:>7.3f}",
              flush=True)

    print("\nAPD-basis reference (same target, LIFE=10/RAMP=0.6): "
          "active 0.71 inject 0.94 purity 1.00 xlayer 1.00 keep_only 0.108 (off 0.639)", flush=True)
    print("APD-basis reference (LIFE=0):                          "
          "active 1.00 inject 0.25 purity 0.99 xlayer 1.00 keep_only 0.060 (off 0.600)", flush=True)
    print("VPD_CLUSTER DONE", flush=True)


if __name__ == "__main__":
    main()
