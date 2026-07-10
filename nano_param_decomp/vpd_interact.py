"""Poke the VPD (rank-1 atom) components of attn-only-2l — the control for interact_apd.py.
If VPD's atoms are ALSO dense/meaningless blobs on this model, the problem is the model/setup, not
APD. If VPD localizes induction to the known heads (prev-token (0,3), induction (1,6)), APD is behind.

Loads the saved VPD decomposition (run.decompose format) and reports, on an in-distribution induction
probe: (1) top induction-CI atoms and which (layer,head) they map to, (2) keep-only-top-k induction
copy (localization/sparsity), (3) per-HEAD causal ablation — zero all atoms mapping to a head, measure
induction drop; the induction circuit should be the heads whose ablation hurts most.

Run:  CUDA_VISIBLE_DEVICES=0 python -m nano_param_decomp.vpd_interact
"""

import os

os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"

import torch
from torch import Tensor

from . import run
from .run import clear_wrapper_masks, induction_copy_acc, install_components, kl_logits, set_wrapper_masks
from .attn_only_2l import INDUCTION_HEAD, PREV_TOKEN_HEAD, load_attn_only_2l_target

CKPT = "/tmp/attn2l_compare/vpd.pt"
POOL = "/tmp/attn2l_compare/pool.pt"


@torch.no_grad()
def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seq_len = int(os.environ.get("SEQ", "128"))
    topk = int(os.environ.get("TOPK", "12"))
    ck = torch.load(CKPT, weights_only=False)
    cfg = run.Config(**dict(ck["cfg"]))
    model = load_attn_only_2l_target().to(device)
    d_head = model.blocks[0].d_head
    n_heads = model.blocks[0].n_heads
    wrappers = install_components(model, ck["C_per_module"])
    model = model.to(device)
    for name, w in wrappers.items():
        w.V.data = ck["wrappers"][name]["V"].to(device)
        w.U.data = ck["wrappers"][name]["U"].to(device)
    d_in = {n: int(w.W_target.shape[1]) for n, w in wrappers.items()}
    ci_fn = run.CITransformer(d_in, cfg.C_per_module, cfg).to(device).eval()
    ci_fn.load_state_dict(ck["ci_fn"])
    names = sorted(wrappers)
    print(f"loaded VPD: {sum(cfg.C_per_module.values())} atoms over {len(wrappers)} modules; "
          f"answer key prev-token {PREV_TOKEN_HEAD}, induction {INDUCTION_HEAD}\n", flush=True)

    # each atom -> (layer, head): q/k/v write via U[c,:] (d_out=heads); o reads via V[:,c] (d_in=heads)
    def atom_head(name: str, c: int) -> int:
        w = wrappers[name]
        vec = w.V[:, c] if name.endswith("o_proj") else w.U[c, :]  # [512]
        per = vec.view(n_heads, d_head).pow(2).sum(dim=1)  # [n_heads]
        return int(per.argmax())

    def layer_of(name: str) -> int:
        return 0 if "blocks.0" in name else 1

    # induction probe (in-distribution)
    pool = torch.load(POOL, weights_only=True)[:, :seq_len].to(device)
    L = seq_len // 2
    first = pool[:32, :L]
    seq = torch.cat([first, first], dim=1)
    B, S = seq.shape

    clear_wrapper_masks(wrappers)
    target = model(seq)
    acts = {n: w.last_input for n, w in wrappers.items()}
    ci_lower, ci_upper, _ = ci_fn(acts)  # dict module -> [B,S,C_m]
    base_copy = induction_copy_acc(target, first, L)

    zeros = {n: torch.zeros(B, S, device=device) for n in wrappers}
    ci_all = _fwd(model, wrappers, seq, ci_lower, zeros)
    base_ci_copy = induction_copy_acc(ci_all, first, L)
    print(f"induction copy: original={base_copy:.3f}  ci-masked(all atoms)={base_ci_copy:.3f}", flush=True)

    # global atom ranking by induction-CI (mean upper-CI on 2nd-copy positions)
    atoms = []  # (imp, name, c, layer, head)
    for n in names:
        imp = ci_upper[n][:, L:, :].mean(dim=(0, 1))  # [C_m]
        for c in range(imp.shape[0]):
            atoms.append((imp[c].item(), n, c, layer_of(n), atom_head(n, c)))
    atoms.sort(key=lambda t: -t[0])

    print(f"\n=== top {topk} induction-CI atoms (which head do they live in?) ===", flush=True)
    from collections import Counter
    head_counter = Counter()
    for imp, n, c, l, h in atoms[:topk]:
        proj = n.split(".")[-1]
        print(f"  ci={imp:.3f}  {n}:{c}  -> L{l}H{h} ({proj})", flush=True)
        head_counter[(l, h)] += 1
    on_known = sum(head_counter[hd] for hd in (PREV_TOKEN_HEAD, INDUCTION_HEAD))
    print(f"  of top {topk}: {on_known} on known heads {PREV_TOKEN_HEAD}/{INDUCTION_HEAD}; "
          f"head spread: {dict(head_counter)}", flush=True)

    # keep-only top-k atoms across modules (localization/sparsity)
    print(f"\n=== keep-only top-k induction-CI atoms -> induction copy ===", flush=True)
    for k in [1, 3, 8, 16, 32, 64]:
        masks = {n: torch.zeros(B, S, cfg.C_per_module[n], device=device) for n in wrappers}
        for imp, n, c, l, h in atoms[:k]:
            masks[n][..., c] = 1.0
        copy = induction_copy_acc(_fwd(model, wrappers, seq, masks, zeros), first, L)
        print(f"  top {k:>3} atoms -> induction copy {copy:.3f}", flush=True)

    # per-HEAD causal ablation: zero all atoms mapping to head (l,h), keep rest at ci
    print(f"\n=== per-head ablation (zero that head's atoms) -> induction drop from {base_ci_copy:.3f} ===", flush=True)
    head_to_atoms: dict[tuple[int, int], dict[str, list[int]]] = {}
    for imp, n, c, l, h in atoms:
        head_to_atoms.setdefault((l, h), {}).setdefault(n, []).append(c)
    drops = []
    for (l, h), mods in head_to_atoms.items():
        masks = {n: ci_lower[n].clone() for n in wrappers}
        for n, cs in mods.items():
            masks[n][..., cs] = 0.0
        copy = induction_copy_acc(_fwd(model, wrappers, seq, masks, zeros), first, L)
        drops.append((base_ci_copy - copy, (l, h)))
    drops.sort(reverse=True)
    for drop, (l, h) in drops[:8]:
        tag = " <-- PREV-TOKEN" if (l, h) == PREV_TOKEN_HEAD else (" <-- INDUCTION" if (l, h) == INDUCTION_HEAD else "")
        print(f"  ablate L{l}H{h}: induction drop {drop:+.3f}{tag}", flush=True)
    print("VPD_INTERACT DONE", flush=True)


def _fwd(model, wrappers, seq, masks, deltas):
    set_wrapper_masks(wrappers, masks, deltas, routing=None)
    try:
        return model(seq)
    finally:
        clear_wrapper_masks(wrappers)


if __name__ == "__main__":
    main()
