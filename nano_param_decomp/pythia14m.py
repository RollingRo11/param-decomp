"""Target loader + data for EleutherAI/pythia-14m, an induction-capable model comparable in
decomposition scale to SS2L (~6912 atoms vs 7104) but 6 layers and Pile-trained (so it actually
inducts: ~0.76 copy on repeated random tokens, vs SS2L's 0.0).

GPTNeoX is all `nn.Linear`, so the per-layer attention/mlp matrices drop straight into
`run.install_components` by path. We decompose those (not the embeddings)."""

import os

os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"

import types
from collections.abc import Iterator

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

MODEL_ID = "EleutherAI/pythia-14m"
N_LAYERS = 6

# 4 decomposed matrices per layer; values are output dims (= atom counts). 1152/layer x 6 = 6912.
C_PER_MODULE_PYTHIA_14M: dict[str, int] = {
    f"gpt_neox.layers.{i}.{sub}": c
    for i in range(N_LAYERS)
    for sub, c in (
        ("attention.query_key_value", 384),
        ("attention.dense", 128),
        ("mlp.dense_h_to_4h", 512),
        ("mlp.dense_4h_to_h", 128),
    )
}


def load_pythia14m_target() -> nn.Module:
    """Frozen pythia-14m whose `forward(idx)` returns bare logits (matching the SS2L target API)."""
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(MODEL_ID)
    original_forward = model.forward

    def forward_logits_only(_self: nn.Module, idx: Tensor) -> Tensor:
        return original_forward(input_ids=idx).logits

    model.forward = types.MethodType(forward_logits_only, model)
    model.eval()
    return model


@torch.no_grad()
def generate_pool(
    model: nn.Module, n_seqs: int, seq_len: int, device: torch.device, seed: int = 0,
    temperature: float = 1.0,
) -> Tensor:
    """Autoregressively sample a fixed pool from the frozen target (in-distribution, no dataset)."""
    model = model.to(device).eval()
    vocab = model.config.vocab_size
    g = torch.Generator(device=device).manual_seed(seed)
    chunks: list[Tensor] = []
    bs, done = 64, 0
    while done < n_seqs:
        b = min(bs, n_seqs - done)
        seq = torch.randint(0, vocab, (b, 1), device=device, generator=g)
        for _ in range(seq_len - 1):
            logits = model(seq)[:, -1, :]
            probs = F.softmax(logits / temperature, dim=-1)
            nxt = torch.multinomial(probs, 1, generator=g)
            seq = torch.cat([seq, nxt], dim=1)
        chunks.append(seq.cpu())
        done += b
    return torch.cat(chunks, dim=0)[:n_seqs]


def pool_loader(pool: Tensor, batch_size: int, seed: int) -> Iterator[Tensor]:
    g = torch.Generator().manual_seed(seed)
    n = pool.shape[0]
    while True:
        idx = torch.randint(0, n, (batch_size,), generator=g)
        yield pool[idx]


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_pythia14m_target().to(device)
    n_atoms = sum(C_PER_MODULE_PYTHIA_14M.values())
    print(f"pythia-14m loaded | {n_atoms} atoms over {len(C_PER_MODULE_PYTHIA_14M)} modules", flush=True)
    # sanity: all decomposition targets resolve and are nn.Linear
    for path in C_PER_MODULE_PYTHIA_14M:
        mod = model.get_submodule(path)
        assert isinstance(mod, nn.Linear), f"{path} is {type(mod)}"
        assert mod.weight.shape[0] == C_PER_MODULE_PYTHIA_14M[path], path
    pool = generate_pool(model, 4, 32, device, seed=0)
    print("pool:", tuple(pool.shape), "max id:", int(pool.max()), flush=True)
    print("PYTHIA14M OK", flush=True)
