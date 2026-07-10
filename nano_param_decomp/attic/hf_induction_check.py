"""Test whether an arbitrary HuggingFace causal LM does induction (repeated-random-token copy test),
to pick an online induction-capable model comparable to SS2L. Runs on CPU. Models via env MODELS
(comma-separated). Reports params, layers, 2nd-copy copy accuracy, and P(induction token)."""

import os

os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"

import torch
import torch.nn.functional as F

MODELS = os.environ.get("MODELS", "EleutherAI/pythia-70m,EleutherAI/pythia-160m").split(",")
L = int(os.environ.get("L", "64"))
N = int(os.environ.get("N", "32"))


@torch.no_grad()
def check(model_id: str) -> None:
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(model_id)
    model.eval()
    n_params = sum(p.numel() for p in model.parameters())
    n_layers = getattr(model.config, "num_hidden_layers", getattr(model.config, "n_layer", "?"))
    vocab = model.config.vocab_size
    g = torch.Generator().manual_seed(0)
    first = torch.randint(0, vocab, (N, L), generator=g)
    seq = torch.cat([first, first], dim=1)  # [N, 2L]
    logits = model(seq).logits  # [N, 2L, vocab]
    pred = logits[:, :-1].argmax(dim=-1)
    tgt = seq[:, 1:]
    correct = (pred == tgt).float()
    first_acc = correct[:, : L - 1].mean().item()
    second_acc = correct[:, L:].mean().item()
    logp = F.log_softmax(logits, dim=-1)
    pos = torch.arange(L, 2 * L - 1)
    ind_tok = first[:, 1:L]
    lp = logp[:, pos, :].gather(-1, ind_tok.unsqueeze(-1)).squeeze(-1)
    print(
        f"[{model_id}] params={n_params/1e6:.1f}M layers={n_layers} vocab={vocab} | "
        f"1st-copy acc={first_acc:.3f} 2nd-copy acc={second_acc:.3f} "
        f"P(ind tok|2nd)={lp.exp().mean().item():.3f} lift={second_acc-first_acc:+.3f}",
        flush=True,
    )


def main() -> None:
    for m in MODELS:
        try:
            check(m.strip())
        except Exception as e:  # noqa: BLE001  -- want to test the rest even if one fails
            print(f"[{m.strip()}] FAILED: {type(e).__name__}: {e}", flush=True)
    print("HF INDUCTION CHECK DONE")


if __name__ == "__main__":
    main()
