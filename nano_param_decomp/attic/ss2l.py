"""Keyless loader for the 2-layer SimpleStories target + its data, for the matryoshka-vs-VPD
comparison. Files were pre-downloaded anonymously from the public `goodfire/spd` W&B project to
`out/pretrain_cache/spd-gf6rbga0/` (model_config.yaml, model_step_99999.pt, tokenizer.json).

The target is the canonical 2-layer LlamaSimpleMLP (n_embd 192, 2 layers, vocab 4019) — a small
transformer with a real cross-layer structure, so matryoshka's cross-layer components are testable.
"""

import os

# The box sets HF_HUB_ENABLE_HF_TRANSFER=1 globally but hf_transfer isn't installed; disable it
# before any huggingface import so dataset/tokenizer downloads fall back to plain HTTP.
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"

import types
from collections.abc import Iterator

import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from torch import Tensor

CACHE = os.path.join(os.path.dirname(__file__), "..", "out", "pretrain_cache", "spd-gf6rbga0")

# 6 module types x 2 layers; mirrors simplestories_2L.C_PER_MODULE_SS_2L (atom counts unchanged).
C_PER_MODULE_SS_2L: dict[str, int] = {
    "h.0.attn.q_proj": 288,
    "h.0.attn.k_proj": 288,
    "h.0.attn.v_proj": 384,
    "h.0.attn.o_proj": 480,
    "h.0.mlp.c_fc": 1152,
    "h.0.mlp.down_proj": 960,
    "h.1.attn.q_proj": 288,
    "h.1.attn.k_proj": 288,
    "h.1.attn.v_proj": 384,
    "h.1.attn.o_proj": 480,
    "h.1.mlp.c_fc": 1152,
    "h.1.mlp.down_proj": 960,
}


def load_ss2l_target() -> nn.Module:
    """Build the 2-layer LlamaSimpleMLP from the cached config + weights (no W&B key). Returns a
    fresh, frozen model whose `forward(idx)` yields bare logits. Call once per decomposition
    (install_components mutates the model in place)."""
    from param_decomp_lab.experiments.lm.pretrain.models.llama_simple_mlp import (
        LlamaSimpleMLP,
        LlamaSimpleMLPConfig,
    )

    with open(os.path.join(CACHE, "model_config.yaml")) as f:
        cfg_d = yaml.safe_load(f)
    cfg_d.setdefault("model_type", "LlamaSimpleMLP")
    try:
        model_cfg = LlamaSimpleMLPConfig(**cfg_d)
    except Exception:
        cfg_d.pop("model_type", None)
        model_cfg = LlamaSimpleMLPConfig(**cfg_d)
    model = LlamaSimpleMLP(model_cfg)

    sd = torch.load(
        os.path.join(CACHE, "model_step_99999.pt"), map_location="cpu", weights_only=True
    )
    sd = sd["model"] if isinstance(sd, dict) and "model" in sd and hasattr(sd["model"], "keys") else sd
    model.load_state_dict(sd)

    original_forward = model.forward

    def forward_logits_only(_self: nn.Module, idx: Tensor) -> Tensor:
        logits, _loss = original_forward(idx)
        assert logits is not None
        return logits

    model.forward = types.MethodType(forward_logits_only, model)
    model.eval()
    return model


def _tokenizer():
    from transformers import PreTrainedTokenizerFast

    return PreTrainedTokenizerFast(tokenizer_file=os.path.join(CACHE, "tokenizer.json"))


def make_loader(batch_size: int, seq_len: int, seed: int, split: str = "train") -> Iterator[Tensor]:
    """Stream SimpleStories, tokenize on the fly with the matching tokenizer, EOS-pack into
    fixed `seq_len` chunks. Single-process (no rank sharding — this comparison runs on 1 GPU)."""
    import datasets

    tok = _tokenizer()
    eos = tok.eos_token_id if tok.eos_token_id is not None else 0
    ds = datasets.load_dataset("SimpleStories/SimpleStories", split=split, streaming=True)
    ds = ds.shuffle(seed=seed, buffer_size=1000)
    while True:
        buf: list[int] = []
        batch: list[Tensor] = []
        for ex in ds:
            buf.extend(tok.encode(ex["story"].lower()))
            buf.append(eos)
            while len(buf) >= seq_len:
                batch.append(torch.tensor(buf[:seq_len], dtype=torch.long))
                buf = buf[seq_len:]
                if len(batch) == batch_size:
                    yield torch.stack(batch, dim=0)
                    batch = []


@torch.no_grad()
def generate_pool(
    model: nn.Module, n_seqs: int, seq_len: int, device: torch.device, seed: int = 0,
    temperature: float = 1.0,
) -> Tensor:
    """Autoregressively sample a fixed pool of sequences from the (frozen) target. This is the
    decomposition's input data: exactly in-distribution for the target, no dataset download.
    Returns [n_seqs, seq_len] int64 on CPU."""
    model = model.to(device).eval()
    vocab = model.lm_head.weight.shape[0]
    g = torch.Generator(device=device).manual_seed(seed)
    chunks: list[Tensor] = []
    bs = 128
    done = 0
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
    # Sanity: target loads, generates, and assigns low CE to its own samples (trained, not random).
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_ss2l_target().to(device)
    vocab = model.lm_head.weight.shape[0]
    n_params = sum(p.numel() for p in model.parameters())
    print(f"target loaded: {n_params:,} params, vocab {vocab} (uniform CE would be {torch.log(torch.tensor(float(vocab))):.2f})", flush=True)
    pool = generate_pool(model, n_seqs=16, seq_len=128, device=device, seed=0)
    print("sampled pool:", tuple(pool.shape), "max token id:", int(pool.max()), flush=True)
    batch = pool.to(device)
    with torch.no_grad():
        logits = model(batch)
    ce = F.cross_entropy(logits[:, :-1].reshape(-1, logits.shape[-1]), batch[:, 1:].reshape(-1))
    print(f"logits shape: {tuple(logits.shape)} | next-token CE on self-samples: {ce.item():.3f}")
    print("SS2L OK")
