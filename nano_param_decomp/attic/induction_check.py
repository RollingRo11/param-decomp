"""Does the SS2L target do induction? Classic test: feed a sequence of RANDOM tokens repeated
twice ([r_0..r_{L-1}, r_0..r_{L-1}]). At position L+i the induction-correct next token is
r_{(i+1) mod L} (what followed r_i the first time). Induction accuracy = fraction of second-copy
positions whose argmax matches that. Compare to first-copy accuracy (no induction possible) as a
baseline. Runs on CPU to avoid disturbing GPU runs.
"""

import os

os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"

import torch
import torch.nn.functional as F

from .ss2l import generate_pool, load_ss2l_target


@torch.no_grad()
def induction_scores(model, vocab, L, n_seq, device, seed, src):
    g = torch.Generator(device=device).manual_seed(seed)
    if src == "random":
        first = torch.randint(0, vocab, (n_seq, L), device=device, generator=g)
    else:  # in-distribution: take the first L tokens of model-sampled sequences
        pool = generate_pool(model, n_seq, L, device, seed=seed).to(device)
        first = pool[:, :L]
    seq = torch.cat([first, first], dim=1)  # [n_seq, 2L]
    logits = model(seq)  # [n_seq, 2L, vocab]
    pred = logits.argmax(dim=-1)  # [n_seq, 2L]

    # target[t] = seq[t+1]; we score predictions at positions t against the true next token.
    tgt = seq[:, 1:]  # [n_seq, 2L-1]
    pred = pred[:, :-1]
    correct = (pred == tgt).float()  # [n_seq, 2L-1]

    # first copy: positions 0..L-2 ; second copy (induction-eligible): positions L..2L-2
    first_acc = correct[:, : L - 1].mean().item()
    second_acc = correct[:, L:].mean().item()

    # also: rank/prob the model assigns to the induction-correct token in the 2nd copy
    logp = F.log_softmax(logits, dim=-1)
    # at position L+i (i in 0..L-2) the induction-correct next token is first[:, i+1]
    pos = torch.arange(L, 2 * L - 1, device=device)
    ind_tok = first[:, 1:L]  # [n_seq, L-1]
    lp = logp[:, pos, :].gather(-1, ind_tok.unsqueeze(-1)).squeeze(-1)  # [n_seq, L-1]
    ind_token_prob = lp.exp().mean().item()
    return first_acc, second_acc, ind_token_prob


def main():
    device = torch.device("cpu")
    model = load_ss2l_target().to(device)
    vocab = model.lm_head.weight.shape[0]
    for src in ("random", "indist"):
        fa, sa, p = induction_scores(model, vocab, L=64, n_seq=64, device=device, seed=0, src=src)
        print(
            f"[{src:7}] 1st-copy next-tok acc={fa:.3f}  2nd-copy acc={sa:.3f}  "
            f"P(induction tok | 2nd copy)={p:.3f}  lift={sa - fa:+.3f}",
            flush=True,
        )
    print("INDUCTION CHECK DONE")


if __name__ == "__main__":
    main()
