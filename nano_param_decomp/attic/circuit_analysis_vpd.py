"""Manual examination of the induction circuit VPD extracted. Loads vpd_s100000.pt, builds an
in-distribution repeated probe, and localizes induction by (1) per-module causal importance at the
2nd-copy positions and (2) per-module ablation (zero a module's components, measure the induction
copy-accuracy drop). The modules whose ablation hurts induction most ARE the circuit."""

import os

os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"

import torch

from . import run
from .run import _require, clear_wrapper_masks, induction_copy_acc, install_components, set_wrapper_masks
from .pythia14m import load_pythia14m_target

CKPT = "/tmp/pythia_compare/vpd_s100000.pt"
POOL = "/tmp/pythia_compare/pool.pt"
L = 64


def main() -> None:
    device = torch.device("cpu")
    ck = torch.load(CKPT, weights_only=True)
    cfg = run.Config(**dict(ck["cfg"]))
    target = load_pythia14m_target().to(device).float()
    wrappers = install_components(target, ck["C_per_module"])
    for name, w in wrappers.items():
        w.V.data = ck["wrappers"][name]["V"].to(device)
        w.U.data = ck["wrappers"][name]["U"].to(device)
    d_in = {n: int(w.W_target.shape[1]) for n, w in wrappers.items()}
    ci_fn = run.CITransformer(d_in, cfg.C_per_module, cfg).to(device).eval()
    ci_fn.load_state_dict(ck["ci_fn"])
    names = sorted(wrappers.keys())

    pool = torch.load(POOL, weights_only=True)
    first = pool[int(0.9 * pool.shape[0]):][:32, :L].to(device)
    seq = torch.cat([first, first], dim=1)
    B, S = seq.shape

    @torch.no_grad()
    def ci_of(seq):
        clear_wrapper_masks(wrappers)
        target(seq)
        acts = {n: _require(w.last_input) for n, w in wrappers.items()}
        ci_lower, _u, _p = ci_fn(acts)
        return {n: ci_lower[n].clone() for n in names}

    @torch.no_grad()
    def copy_acc(masks):
        zd = {n: torch.zeros(B, S, device=device) for n in names}
        set_wrapper_masks(wrappers, masks, zd, None)
        try:
            logits = target(seq)
        finally:
            clear_wrapper_masks(wrappers)
        return induction_copy_acc(logits, first, L)

    ci = ci_of(seq)
    base = copy_acc(ci)
    print(f"baseline ci-masked induction copy = {base:.3f}\n", flush=True)

    # 2nd-copy positions are where induction acts; measure CI usage there + ablation drop per module.
    rows = []
    for n in names:
        copy_l0 = (ci[n][:, L:, :] > 0).float().sum(-1).mean().item()  # active atoms/tok at copy pos
        first_l0 = (ci[n][:, :L, :] > 0).float().sum(-1).mean().item()  # ... at first-copy pos
        ablated = {m: (torch.zeros_like(ci[m]) if m == n else ci[m]) for m in names}
        drop = base - copy_acc(ablated)
        rows.append((n, copy_l0, first_l0, drop))

    rows.sort(key=lambda r: -r[3])
    print(f"{'module':<48} {'L0@copy':>8} {'L0@1st':>8} {'abl.drop':>9}")
    print("-" * 78)
    for n, cl0, fl0, drop in rows:
        print(f"{n.replace('gpt_neox.layers.', 'L'):<48} {cl0:>8.2f} {fl0:>8.2f} {drop:>9.3f}", flush=True)

    # Within the single most induction-critical module, the top atoms by mean CI at copy positions.
    top_mod = rows[0][0]
    mean_ci_copy = ci[top_mod][:, L:, :].mean(dim=(0, 1))  # [C]
    topk = torch.topk(mean_ci_copy, min(10, mean_ci_copy.numel()))
    print(f"\ntop atoms in {top_mod} by mean CI at copy positions:")
    for v, idx in zip(topk.values.tolist(), topk.indices.tolist()):
        print(f"  atom {idx:4d}  meanCI={v:.3f}", flush=True)

    # Keep-only: zero everything except a candidate subset; how much induction does that subset alone carry?
    def keep_only(pred):
        masks = {m: (ci[m] if pred(m) else torch.zeros_like(ci[m])) for m in names}
        return copy_acc(masks)

    is_attn = lambda m: "attention" in m
    is_mlp = lambda m: "mlp" in m
    L5_qkv = "gpt_neox.layers.5.attention.query_key_value"
    print("\nkeep-only (fraction of baseline 0.263 that the subset alone preserves):")
    print(f"  attention only           {keep_only(is_attn):.3f}")
    print(f"  mlp only                 {keep_only(is_mlp):.3f}")
    print(f"  L4+L5 attention only     {keep_only(lambda m: is_attn(m) and ('layers.4' in m or 'layers.5' in m)):.3f}")
    print(f"  L5 attention only        {keep_only(lambda m: is_attn(m) and 'layers.5' in m):.3f}")

    # Keep only the top-k atoms of L5 qkv (everything else in that module zeroed, rest of model full).
    for k in (3, 5):
        keep_idx = set(torch.topk(mean_ci_copy, k).indices.tolist())
        masks = {m: ci[m].clone() for m in names}
        col_mask = torch.zeros(mean_ci_copy.numel())
        for i in keep_idx:
            col_mask[i] = 1.0
        masks[L5_qkv] = masks[L5_qkv] * col_mask.view(1, 1, -1)
        print(f"  L5-qkv top-{k} atoms only (rest of model full)  {copy_acc(masks):.3f}")
    print("ANALYSIS DONE", flush=True)


if __name__ == "__main__":
    main()
