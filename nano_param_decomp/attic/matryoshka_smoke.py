"""Tiny end-to-end smoke test for matryoshka.py — no downloads, runs a few steps on a toy model.

    python -m nano_param_decomp.matryoshka_smoke
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .matryoshka import Config, decompose

D = 32
VOCAB = 50


class Attn(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.q_proj = nn.Linear(D, D, bias=False)
        self.k_proj = nn.Linear(D, D, bias=False)
        self.v_proj = nn.Linear(D, D, bias=False)
        self.o_proj = nn.Linear(D, D, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        q, k, v = self.q_proj(x), self.k_proj(x), self.v_proj(x)
        gate = torch.sigmoid((q * k).sum(-1, keepdim=True))
        return self.o_proj(v * gate)


class Mlp(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.c_fc = nn.Linear(D, 4 * D, bias=False)
        self.down_proj = nn.Linear(4 * D, D, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        return self.down_proj(F.gelu(self.c_fc(x)))


class Block(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.attn = Attn()
        self.mlp = Mlp()

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.attn(x)
        return x + self.mlp(x)


class ToyLM(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embed = nn.Embedding(VOCAB, D)
        self.h = nn.ModuleList([Block()])
        self.unembed = nn.Linear(D, VOCAB, bias=False)

    def forward(self, idx: Tensor) -> Tensor:
        x = self.embed(idx)
        for block in self.h:
            x = block(x)
        return self.unembed(x)


C_PER_MODULE = {
    "h.0.attn.q_proj": 8,
    "h.0.attn.k_proj": 8,
    "h.0.attn.v_proj": 8,
    "h.0.attn.o_proj": 8,
    "h.0.mlp.c_fc": 16,
    "h.0.mlp.down_proj": 16,
}


def loader(batch: int, seq: int):
    g = torch.Generator().manual_seed(0)
    while True:
        yield torch.randint(0, VOCAB, (batch, seq), generator=g)


if __name__ == "__main__":
    cfg = Config(
        C_per_module=C_PER_MODULE,
        n_steps=3,
        batch_size=2,
        seq_len=8,
        faithfulness_warmup_steps=2,
        ci_d_model=16,
        ci_n_blocks=1,
        ci_n_heads=2,
        ci_mlp_hidden=32,
        eval_freq=2,
        eval_batch_size=2,
        log_every=1,
        n_components=4,
        tau_start=2.0,
        tau_end=0.5,
        ppgd_inner_steps=2,
        use_wandb=False,
    )
    decompose(ToyLM(), cfg, loader(cfg.batch_size, cfg.seq_len), loader(cfg.eval_batch_size, cfg.seq_len))
    print("SMOKE OK")
