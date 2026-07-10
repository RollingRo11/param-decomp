"""Induction analysis of the VPD paper's OWN pile-4L decomposition (goodfire/spd/s-55ea3f9b,
model_400000.pth, 38,912 subcomponents, 400k steps) — the same population-level questions we asked
of our Pythia decomposition:

  1. Which heads of the target are induction heads (prefix-matching attention score)?
  2. At copy events, which subcomponents are active, and how consistently (P(active|copy))?
  3. Who OWNS the induction heads' weights (per-atom share of head-slice energy), and are the
     owners the same atoms that activate on copying? Is there a fixed induction crew?

Loading bypasses the current PDConfig schema (the run predates it): we validate only the ci_config,
resolve decomposition targets from the legacy `module_info`, and load the state dict directly.

Run:  CUDA_VISIBLE_DEVICES=0 python -m nano_param_decomp.pile4l_induction
Env:  B (default 48), L (default 64)
"""

import os
import sys
import types

import torch
import yaml

CKPT_DIR = "out/pile4l"


def load_pile4l_decomposition(device):
    stub = types.ModuleType("param_decomp_lab.infra.hf_http")
    stub.configure_hf_http_retries = lambda *a, **k: None
    sys.modules["param_decomp_lab.infra.hf_http"] = stub

    from pydantic import TypeAdapter
    from param_decomp.configs import PDConfig
    from param_decomp.decomposition_targets import (
        DecompositionTargetConfig,
        resolve_decomposition_targets,
    )
    from param_decomp.component_model import ComponentModel
    from .pile_4L import load_paper_target_model

    raw = yaml.safe_load(open(os.path.join(CKPT_DIR, "final_config.yaml")))
    ci_raw = {k: v for k, v in raw["ci_config"].items() if v is not None}
    ci_config = TypeAdapter(PDConfig.model_fields["ci_config"].annotation).validate_python(ci_raw)
    targets = [DecompositionTargetConfig(module_pattern=m["module_pattern"], C=m["C"])
               for m in raw["module_info"]]
    target = load_paper_target_model()
    target.eval(); target.requires_grad_(False)
    resolved = resolve_decomposition_targets(target, targets)
    cm = ComponentModel(target_model=target, run_batch=lambda m, b: m(b),
                        decomposition_targets=resolved, ci_config=ci_config,
                        sigmoid_type=raw["sigmoid_type"])
    sd = torch.load(os.path.join(CKPT_DIR, "model_400000.pth"), map_location="cpu", weights_only=True)
    cm.load_state_dict(sd)
    return cm.to(device), target.to(device)


@torch.no_grad()
def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    B = int(os.environ.get("B", "48"))
    L = int(os.environ.get("L", "64"))
    cm, target = load_pile4l_decomposition(device)
    print("loaded paper decomposition:",
          sum(c.V.shape[-1] if c.V.dim() == 2 else 0 for c in cm.components.values()), "atoms-ish;",
          len(cm.components), "modules", flush=True)

    # --- pool: sample from the frozen target itself ---
    pool_path = os.path.join(CKPT_DIR, "pool.pt")
    vocab = target.transformer.wte.weight.shape[0] if hasattr(target, "transformer") else None
    if os.path.exists(pool_path):
        pool = torch.load(pool_path, weights_only=True).to(device)
    else:
        g = torch.Generator(device=device).manual_seed(0)
        if vocab is None:  # fall back: find embedding
            emb = next(m for m in target.modules() if isinstance(m, torch.nn.Embedding))
            vocab = emb.num_embeddings
        seqs = torch.randint(0, vocab, (B, 1), device=device, generator=g)
        for _ in range(L - 1):
            logits = target(seqs)[:, -1, :]
            nxt = torch.multinomial(torch.softmax(logits, -1), 1, generator=g)
            seqs = torch.cat([seqs, nxt], 1)
        pool = seqs
        torch.save(pool.cpu(), pool_path)
    pool = pool[:B, :L]
    seq = torch.cat([pool, pool], dim=1)  # [B, 2L]
    print(f"probe {tuple(seq.shape)}", flush=True)

    # --- 1. induction heads from attention patterns (hook q/k, replicate module rotary) ---
    attn_mods = [(n, m) for n, m in target.named_modules() if type(m).__name__ == "CausalSelfAttention"]
    t2 = torch.arange(L + 1, 2 * L, device=device)
    head_scores = {}
    qk_cache = {}

    def mk_hook(name):
        def hook(mod, inp, out):
            qk_cache[name] = (mod, inp[0].detach())
        return hook

    handles = [m.register_forward_hook(mk_hook(n)) for n, m in attn_mods]
    target(seq[:16])
    for h in handles:
        h.remove()
    for lname, (mod, x) in qk_cache.items():
        Bq, T, _ = x.shape
        if mod.use_grouped_query_attention:
            q = mod.q_proj(x); k = mod.k_proj(x)
        else:
            q, k, _ = mod.c_attn(x).split(mod.n_embd, dim=2)
        q = q.view(Bq, T, mod.n_head, mod.head_dim).transpose(1, 2)
        kv_heads = getattr(mod, "n_key_value_heads", mod.n_head)
        k = k.view(Bq, T, kv_heads, mod.head_dim).transpose(1, 2)
        cos = mod.rotary_cos[:T].unsqueeze(0); sin = mod.rotary_sin[:T].unsqueeze(0)
        q, k = mod.apply_rotary_pos_emb(q, k, cos, sin)
        if kv_heads != mod.n_head:
            k = k.repeat_interleave(mod.n_head // kv_heads, dim=1)
        att = (q @ k.transpose(-2, -1)) / mod.head_dim ** 0.5
        mask = torch.tril(torch.ones(T, T, device=device, dtype=torch.bool))
        att = att.masked_fill(~mask, float("-inf")).softmax(-1)
        layer = int(lname.split(".")[1]) if lname.startswith("h.") else int(lname.split(".h.")[-1].split(".")[0])
        for h in range(mod.n_head):
            head_scores[(layer, h)] = att[:, h, t2, t2 - L + 1].mean().item()
    top = sorted(head_scores.items(), key=lambda kv: -kv[1])
    print("\ntop heads by induction attention score:", flush=True)
    for (l, h), s in top[:6]:
        print(f"  L{l}H{h}: {s:.3f}", flush=True)
    ind_heads = [(l, h) for (l, h), s in top if s > 0.2] or [top[0][0]]
    print(f"induction heads (score>0.2): {ind_heads}", flush=True)

    # --- 2. gates on the probe + copy events ---
    pre_acts = {}
    mods = {n: target.get_submodule(n) for n in cm.components}
    def mk_in_hook(name):
        def hook(mod, inp, out):
            pre_acts.setdefault(name, []).append(inp[0].detach())
        return hook
    handles = [m.register_forward_hook(mk_in_hook(n)) for n, m in mods.items()]
    preds = []
    for lo in range(0, B, 16):
        logits = target(seq[lo:lo + 16])
        preds.append(logits.argmax(-1))
    for h in handles:
        h.remove()
    pred = torch.cat(preds)
    acts = {n: torch.cat(v) for n, v in pre_acts.items()}
    ci_lower = cm.calc_causal_importances(acts, sampling="continuous").lower_leaky
    t1 = torch.arange(1, L - 1, device=device); t2b = t1 + L
    tgt_tok = seq.gather(1, (t1 + 1).expand(B, -1))
    copy_event = (pred[:, t2b] == tgt_tok) & (pred[:, t1] != tgt_tok)
    print(f"\ncopy events: {int(copy_event.sum())}", flush=True)

    # --- 3. ownership + consistency per induction head ---
    d_head = attn_mods[0][1].head_dim
    print(f"\n{'head/module':<28} {'atom':>5} {'slice share':>12} {'P(act|copy)':>12} {'rate':>7}", flush=True)
    for (l, h) in ind_heads:
        for n, comp in cm.components.items():
            if not n.startswith(f"h.{l}.") or "attn" not in n:
                continue
            mod = mods[n]
            W = mod.weight  # [d_out, d_in]; atom c = outer(U[c, :] rows, V[:, c] cols)
            U, V = comp.U, comp.V  # U [C, d_out], V [d_in, C]
            sl = slice(h * d_head, (h + 1) * d_head)
            if "o_proj" in n:  # heads index INPUT columns
                slice_tot = W[:, sl].pow(2).sum()
                atom_mass = V[sl, :].pow(2).sum(0) * U.pow(2).sum(1)
            else:              # q/k/v: heads index OUTPUT rows
                if (h + 1) * d_head > W.shape[0]:
                    continue
                slice_tot = W[sl, :].pow(2).sum()
                atom_mass = U[:, sl].pow(2).sum(1) * V.pow(2).sum(0)
            share = atom_mass / slice_tot
            gl = ci_lower[n]
            act = gl[:, t2b, :] > 0.5
            p_copy = act[copy_event].float().mean(0)
            rate = (gl > 0.5).float().mean(dim=(0, 1))
            srt = share.sort(descending=True)
            n50 = int((srt.values.cumsum(0) < 0.5).sum()) + 1
            tag = f"L{l}H{h} {n.split('.')[-1]}"
            for c in srt.indices[:2].tolist():
                print(f"{tag:<28} {c:>5} {share[c]:>12.1%} {p_copy[c]:>12.2f} {rate[c]:>7.3f}", flush=True)
            print(f"{tag:<28}   -> atoms for 50% of slice: {n50}", flush=True)
    print("\nPILE4L_INDUCTION DONE", flush=True)


if __name__ == "__main__":
    main()
