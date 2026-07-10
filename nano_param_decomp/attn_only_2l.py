"""Target loader + data for Neel Nanda's `attn-only-2l` — the canonical model whose induction
circuit is published: a previous-token head in layer 0 feeding an induction head in layer 1
(Elhage et al. 2021, Olsson et al. 2022).

TransformerLens stores attention as per-head tensors (W_Q [n_heads, d_model, d_head], etc.) and
uses shortformer positional embeddings (position added to the Q/K inputs only, never to V or the
residual stream). The decomposition pipeline hooks `nn.Linear` modules, so we reimplement the
forward as a plain module with q/k/v/o `nn.Linear` projections, loading the TL-*processed* weights
(LN folded, writing/unembed centered, value biases folded). `load_attn_only_2l_target` returns the
plain model; `_tl_logits_match` validates it reproduces TL logits before any training."""

import os

os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"

from collections.abc import Iterator

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

TL_NAME = "attn-only-2l"
N_LAYERS = 2

# 4 decomposed matrices per layer (q/k/v/o projections), each d_model x d_model = 512 atoms.
# 4 x 512 x 2 layers = 4096 atoms.
C_PER_MODULE_ATTN_ONLY_2L: dict[str, int] = {
    f"blocks.{i}.{proj}": 512
    for i in range(N_LAYERS)
    for proj in ("q_proj", "k_proj", "v_proj", "o_proj")
}


def _ln_pre(x: Tensor) -> Tensor:
    """TransformerLens LayerNormPre: center then normalize by RMS, no learnable scale/bias."""
    x = x - x.mean(dim=-1, keepdim=True)
    return x / (x.pow(2).mean(dim=-1, keepdim=True) + 1e-5).sqrt()


class AttnBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_head: int) -> None:
        super().__init__()
        self.n_heads = n_heads
        self.d_head = d_head
        self.q_proj = nn.Linear(d_model, n_heads * d_head)
        self.k_proj = nn.Linear(d_model, n_heads * d_head)
        self.v_proj = nn.Linear(d_model, n_heads * d_head)
        self.o_proj = nn.Linear(n_heads * d_head, d_model)

    def forward(self, resid: Tensor) -> Tensor:
        B, S, _ = resid.shape
        normed = _ln_pre(resid)
        q = self.q_proj(normed).view(B, S, self.n_heads, self.d_head)
        k = self.k_proj(normed).view(B, S, self.n_heads, self.d_head)
        v = self.v_proj(normed).view(B, S, self.n_heads, self.d_head)
        scores = torch.einsum("bqhd,bkhd->bhqk", q, k) / (self.d_head**0.5)
        causal = torch.triu(torch.ones(S, S, device=resid.device, dtype=torch.bool), diagonal=1)
        scores = scores.masked_fill(causal[None, None], float("-inf"))
        attn = scores.softmax(dim=-1)
        z = torch.einsum("bhqk,bkhd->bqhd", attn, v).reshape(B, S, self.n_heads * self.d_head)
        return self.o_proj(z)


class AttnOnly2L(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_head: int, n_ctx: int, d_vocab: int) -> None:
        super().__init__()
        self.n_ctx = n_ctx
        self.W_E = nn.Parameter(torch.zeros(d_vocab, d_model))
        self.W_pos = nn.Parameter(torch.zeros(n_ctx, d_model))
        self.blocks = nn.ModuleList(AttnBlock(d_model, n_heads, d_head) for _ in range(N_LAYERS))
        self.W_U = nn.Parameter(torch.zeros(d_model, d_vocab))
        self.b_U = nn.Parameter(torch.zeros(d_vocab))

    def forward(self, idx: Tensor) -> Tensor:
        S = idx.shape[1]
        assert S <= self.n_ctx, f"seq {S} > n_ctx {self.n_ctx}"
        resid = self.W_E[idx] + self.W_pos[:S][None]  # standard learned positional embeddings
        for block in self.blocks:
            resid = resid + block(resid)
        return _ln_pre(resid) @ self.W_U + self.b_U


WEIGHTS_PATH = os.environ.get("ATTN2L_WEIGHTS", "/tmp/pythia_compare/attn_only_2l.pt")

# Published circuit (confirmed by attn_only_induction_check on a [BOS, rand(L), rand(L)] probe):
PREV_TOKEN_HEAD = (0, 3)  # layer 0, head 3   prev-token score ~0.52
INDUCTION_HEAD = (1, 6)  # layer 1, head 6   induction score ~0.68


def _load_from_tl() -> tuple["AttnOnly2L", object]:
    from transformer_lens import HookedTransformer

    tl = HookedTransformer.from_pretrained(TL_NAME)
    c = tl.cfg
    model = AttnOnly2L(c.d_model, c.n_heads, c.d_head, c.n_ctx, c.d_vocab)
    sd = tl.state_dict()
    with torch.no_grad():
        model.W_E.copy_(sd["embed.W_E"])
        model.W_pos.copy_(sd["pos_embed.W_pos"])
        model.W_U.copy_(sd["unembed.W_U"])
        model.b_U.copy_(sd["unembed.b_U"])
        for i, block in enumerate(model.blocks):
            p = f"blocks.{i}.attn."
            # TL: q = einsum('...d,hde->...he', x, W_Q) + b_Q  ->  Linear weight [h*d_head, d_model]
            block.q_proj.weight.copy_(sd[p + "W_Q"].permute(0, 2, 1).reshape(-1, c.d_model))
            block.q_proj.bias.copy_(sd[p + "b_Q"].reshape(-1))
            block.k_proj.weight.copy_(sd[p + "W_K"].permute(0, 2, 1).reshape(-1, c.d_model))
            block.k_proj.bias.copy_(sd[p + "b_K"].reshape(-1))
            block.v_proj.weight.copy_(sd[p + "W_V"].permute(0, 2, 1).reshape(-1, c.d_model))
            block.v_proj.bias.copy_(sd[p + "b_V"].reshape(-1))
            # TL: out = einsum('...he,hed->...d', z, W_O) + b_O  ->  Linear weight [d_model, h*d_head]
            block.o_proj.weight.copy_(sd[p + "W_O"].permute(2, 0, 1).reshape(c.d_model, -1))
            block.o_proj.bias.copy_(sd[p + "b_O"])
    model.eval()
    return model, tl


def export_weights(path: str = WEIGHTS_PATH) -> None:
    """Run once under system python (needs transformer_lens). Caches processed weights + dims so the
    training path loads pure-torch, no TL dependency in the (DDP) venv."""
    model, tl = _load_from_tl()
    c = tl.cfg
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(
        {
            "dims": dict(d_model=c.d_model, n_heads=c.n_heads, d_head=c.d_head,
                         n_ctx=c.n_ctx, d_vocab=c.d_vocab, bos_token_id=tl.tokenizer.bos_token_id),
            "state_dict": model.state_dict(),
        },
        path,
    )
    print(f"saved attn-only-2l weights -> {path}", flush=True)


def load_attn_only_2l_target(path: str = WEIGHTS_PATH) -> nn.Module:
    """Pure-torch load of the cached weights (no transformer_lens). `forward(idx)` returns logits."""
    assert os.path.exists(path), f"missing {path}; run `export_weights` under system python first"
    ck = torch.load(path, weights_only=True)
    d = ck["dims"]
    model = AttnOnly2L(d["d_model"], d["n_heads"], d["d_head"], d["n_ctx"], d["d_vocab"])
    model.load_state_dict(ck["state_dict"])
    model.bos_token_id = d["bos_token_id"]
    model.eval()
    return model


@torch.no_grad()
def generate_pool(
    model: nn.Module, n_seqs: int, seq_len: int, device: torch.device, seed: int = 0,
    temperature: float = 1.0,
) -> Tensor:
    """In-distribution samples: each sequence is seeded with BOS (the model was trained with it)."""
    model = model.to(device).eval()
    bos = int(model.bos_token_id)
    g = torch.Generator(device=device).manual_seed(seed)
    chunks: list[Tensor] = []
    bs, done = 64, 0
    while done < n_seqs:
        b = min(bs, n_seqs - done)
        seq = torch.full((b, 1), bos, device=device, dtype=torch.long)
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


def _validate() -> None:
    """Port must reproduce TL logits, and TL must actually induct (else this target is pointless)."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, tl = _load_from_tl()
    model = model.to(device)
    g = torch.Generator(device=device).manual_seed(0)
    idx = torch.randint(0, tl.cfg.d_vocab, (2, 64), device=device, generator=g)
    with torch.no_grad():
        mine = model(idx)
        theirs = tl(idx, return_type="logits")
    max_abs = (mine - theirs).abs().max().item()
    print(f"port-vs-TL max|Δlogits| = {max_abs:.2e}  (mine {tuple(mine.shape)})", flush=True)
    assert max_abs < 1e-3, "port does not match TransformerLens"

    # induction copy on repeated random tokens
    L = 64
    first = torch.randint(0, tl.cfg.d_vocab, (16, L), device=device, generator=g)
    seq = torch.cat([first, first], dim=1)
    with torch.no_grad():
        logits = model(seq)
    pred = logits[:, L : 2 * L - 1, :].argmax(-1)
    copy = (pred == first[:, 1:L]).float().mean().item()
    print(f"induction copy (repeated random) = {copy:.3f}", flush=True)
    print(f"d_vocab={tl.cfg.d_vocab} d_model={tl.cfg.d_model} n_heads={tl.cfg.n_heads} n_ctx={tl.cfg.n_ctx}", flush=True)
    print("ATTN_ONLY_2L OK", flush=True)


if __name__ == "__main__":
    _validate()
    export_weights()
