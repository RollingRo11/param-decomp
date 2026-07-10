"""Confirm attn-only-2l does induction and identify the induction + previous-token heads via the
canonical attention-pattern scores on a [BOS, rand(L), rand(L)] probe. Run under system python
(needs transformer_lens):  CUDA_VISIBLE_DEVICES="" python3.12 -m nano_param_decomp.attn_only_induction_check
"""

import os

os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"

import torch


def main() -> None:
    from transformer_lens import HookedTransformer

    device = "cpu"
    tl = HookedTransformer.from_pretrained("attn-only-2l").to(device)
    L, B = 64, 16
    g = torch.Generator(device=device).manual_seed(0)
    bos = tl.tokenizer.bos_token_id
    rand = torch.randint(0, tl.cfg.d_vocab, (B, L), device=device, generator=g)
    seq = torch.cat([torch.full((B, 1), bos, device=device), rand, rand], dim=1)  # [B, 2L+1]

    logits, cache = tl.run_with_cache(seq, return_type="logits")

    # Copy accuracy on the 2nd repeat: position (1+L+i) should predict rand[:, i+1].
    pred = logits[:, 1 + L : 1 + 2 * L - 1, :].argmax(-1)
    copy = (pred == rand[:, 1:L]).float().mean().item()
    print(f"induction copy (BOS + repeated random) = {copy:.3f}\n", flush=True)

    # Per-head scores. Induction stripe: 2nd-half dest attends to the token AFTER the previous
    # occurrence of its current token -> src = dest - L + 1. Prev-token stripe: src = dest - 1.
    print(f"{'head':<8} {'induction':>10} {'prev-token':>11}")
    print("-" * 31)
    rows = []
    for layer in range(tl.cfg.n_layers):
        pat = cache["pattern", layer]  # [B, n_heads, S, S]
        for h in range(tl.cfg.n_heads):
            p = pat[:, h]  # [B, S, S]
            dest = torch.arange(1 + L, 1 + 2 * L, device=device)  # 2nd-half destinations
            induction = p[:, dest, dest - L + 1].mean().item()
            prev = p[:, dest, dest - 1].mean().item()
            rows.append((f"{layer}.{h}", induction, prev))

    for name, ind, prev in sorted(rows, key=lambda r: -r[1]):
        flag = "  <- induction" if ind > 0.3 else ("  <- prev-token" if prev > 0.3 else "")
        print(f"{name:<8} {ind:>10.3f} {prev:>11.3f}{flag}", flush=True)
    print("INDUCTION CHECK DONE", flush=True)


if __name__ == "__main__":
    main()
