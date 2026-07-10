"""Toy target models with KNOWN ground-truth mechanisms, for cheaply iterating on VPD vs MPD before
committing to language models. From the APD/SPD papers (Braun et al. 2025; Bushnaq et al. 2025):

  - TMS (Toy Model of Superposition): `x_hat = ReLU(W^T W x + b)`, W in R[n_hidden, n_features].
    Ground truth = one rank-1 mechanism per feature: Z^(c) is W with all columns but c zeroed.
    Single-layer.

  - Residual MLP (compressed computation, and its cross-layer version): a fixed unit-norm embedding
    W_E (W_U = W_E^T) around `n_layers` residual MLP blocks. Trained so each input feature i maps to
    a target function (here y_i = x_i + ReLU(x_i)). With n_layers>1 and the MLP neurons split across
    layers, each feature's computation is DISTRIBUTED ACROSS LAYERS -> the ground-truth mechanism for
    feature i spans both layers' W_in/W_out. This is the case MPD should recover without a separate
    clustering step.

All decomposed matrices are `nn.Linear` so `run.install_components` hooks them directly. Embeddings
(W_E/W_U) are fixed and NOT decomposed, matching the papers."""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


# --- TMS -------------------------------------------------------------------------------------------

class TMS(nn.Module):
    """`x_hat = ReLU(W^T W x + b)` with tied W. We expose the down-projection as an `nn.Linear`
    (`W`, shape [n_hidden, n_features]) so it can be decomposed; the up-projection reuses W^T."""

    def __init__(self, n_features: int, n_hidden: int) -> None:
        super().__init__()
        assert n_features > n_hidden, "superposition needs more features than hidden dims"
        self.n_features = n_features
        self.n_hidden = n_hidden
        self.W = nn.Linear(n_features, n_hidden, bias=False)  # weight [n_hidden, n_features]
        self.b = nn.Parameter(torch.zeros(n_features))
        self.register_buffer("frozen_up", None)  # set by freeze_for_decomposition (W becomes wrapped)

    def _up_weight(self) -> Tensor:
        # tied to W during training; after freezing, a fixed copy so `W` can be decomposed in place
        return self.W.weight.t() if self.frozen_up is None else self.frozen_up

    def forward(self, x: Tensor) -> Tensor:
        h = self.W(x)  # [B, n_hidden]
        out = F.linear(h, self._up_weight())  # = h @ W ; _up_weight() is W^T = [n_features, n_hidden]
        return F.relu(out + self.b)

    def freeze_for_decomposition(self) -> None:
        """Snapshot the up-projection (W^T) so `self.W` can be replaced by a ComponentLinear."""
        self.frozen_up = self.W.weight.t().detach().clone()

    def ground_truth(self) -> Tensor:
        """The mechanisms {Z^(c)} as rank-1 weight matrices for the decomposed `W`: each is W with
        only column c kept. Returned stacked as [n_features, n_hidden, n_features]."""
        W = self.W.weight.detach()  # [n_hidden, n_features]
        gt = torch.zeros(self.n_features, self.n_hidden, self.n_features)
        for c in range(self.n_features):
            gt[c, :, c] = W[:, c]
        return gt


# --- Residual MLP (compressed computation + cross-layer) -------------------------------------------

class ResidMLP(nn.Module):
    """Fixed unit-norm embedding, `n_layers` residual MLP blocks (ReLU), unembed = W_E^T.
    Decomposed matrices: each block's `in_proj` [d_mlp_layer, d_embed] and `out_proj`
    [d_embed, d_mlp_layer]. d_mlp total neurons are split evenly across layers."""

    def __init__(self, n_features: int, d_embed: int, d_mlp: int, n_layers: int, seed: int = 0) -> None:
        super().__init__()
        assert d_mlp % n_layers == 0
        self.n_features = n_features
        self.d_embed = d_embed
        self.n_layers = n_layers
        self.d_mlp_layer = d_mlp // n_layers
        g = torch.Generator().manual_seed(seed)
        E = torch.randn(n_features, d_embed, generator=g)
        E = E / E.norm(dim=1, keepdim=True)  # unit-norm rows
        self.register_buffer("W_E", E)  # [n_features, d_embed], fixed
        self.blocks = nn.ModuleList()
        for _ in range(n_layers):
            blk = nn.ModuleDict({
                "in_proj": nn.Linear(d_embed, self.d_mlp_layer, bias=True),
                "out_proj": nn.Linear(self.d_mlp_layer, d_embed, bias=False),
            })
            self.blocks.append(blk)

    def forward(self, x: Tensor) -> Tensor:
        resid = x @ self.W_E  # [B, d_embed]
        for blk in self.blocks:
            resid = resid + blk["out_proj"](F.relu(blk["in_proj"](resid)))
        return resid @ self.W_E.t()  # unembed -> [B, n_features]


class BoundPairs(nn.Module):
    """A model with genuine MULTIPLICATIVE cross-layer binding (unlike ResidMLP, which is additive).

    Mechanism d = AND of two distinct input features across two layers:
        out_d = ReLU(x[a_d]) * ReLU(x[b_d])
    Layer 0's `proj` row d reads feature a_d; layer 1's `proj` row d reads feature b_d; the forward
    MULTIPLIES the two layers. So atom (layer0, d) and atom (layer1, d) are BOUND: drop either and
    out_d dies. And they do NOT co-activate -- a_d and b_d are different features that fire on
    different inputs -- so co-activation can't group them, only their causal INTERACTION can. Pairs
    are disjoint (feature 2d, 2d+1), so the ground-truth grouping is M cross-layer pairs of 2 atoms."""

    def __init__(self, n_mech: int, seed: int = 0) -> None:
        super().__init__()
        self.n_mech = n_mech
        self.n_features = 2 * n_mech
        self.blocks = nn.ModuleList([
            nn.ModuleDict({"proj": nn.Linear(self.n_features, n_mech, bias=False)}),
            nn.ModuleDict({"proj": nn.Linear(self.n_features, n_mech, bias=False)}),
        ])
        with torch.no_grad():
            w0 = torch.zeros(n_mech, self.n_features)
            w1 = torch.zeros(n_mech, self.n_features)
            for d in range(n_mech):
                w0[d, 2 * d] = 1.0      # layer 0, mech d reads feature 2d
                w1[d, 2 * d + 1] = 1.0  # layer 1, mech d reads feature 2d+1
            self.blocks[0]["proj"].weight.copy_(w0)
            self.blocks[1]["proj"].weight.copy_(w1)

    def forward(self, x: Tensor) -> Tensor:
        h0 = F.relu(self.blocks[0]["proj"](x))  # [B, n_mech]
        h1 = F.relu(self.blocks[1]["proj"](x))
        return h0 * h1  # multiplicative binding

    def mech_active(self, x: Tensor) -> Tensor:
        """[B, n_mech] indicator of which mechanisms fire (both their features present)."""
        return ((x[:, 0::2] > 0) & (x[:, 1::2] > 0)).float()


def target_fn(x: Tensor) -> Tensor:
    """Per-feature target the resid MLP is trained to compute: y_i = x_i + ReLU(x_i)."""
    return x + F.relu(x)


# --- Data ------------------------------------------------------------------------------------------

def feature_batch(n_features: int, batch: int, feature_prob: float, device: torch.device,
                  gen: torch.Generator) -> Tensor:
    """Sparse features: each coord active w.p. `feature_prob`, value ~ U[0,1] when active."""
    active = (torch.rand(batch, n_features, device=device, generator=gen) < feature_prob).float()
    vals = torch.rand(batch, n_features, device=device, generator=gen)
    return active * vals


# --- Target training -------------------------------------------------------------------------------

def train_tms(n_features: int, n_hidden: int, steps: int, batch: int, feature_prob: float,
              lr: float, device: torch.device, seed: int = 0) -> TMS:
    model = TMS(n_features, n_hidden).to(device)
    gen = torch.Generator(device=device).manual_seed(seed)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    for step in range(steps):
        x = feature_batch(n_features, batch, feature_prob, device, gen)
        loss = F.mse_loss(model(x), x)  # TMS reconstructs its (sparse) input
        opt.zero_grad(); loss.backward(); opt.step()
    return model


def train_resid_mlp(n_features: int, d_embed: int, d_mlp: int, n_layers: int, steps: int,
                    batch: int, feature_prob: float, lr: float, device: torch.device,
                    seed: int = 0) -> ResidMLP:
    model = ResidMLP(n_features, d_embed, d_mlp, n_layers, seed=seed).to(device)
    gen = torch.Generator(device=device).manual_seed(seed + 1)
    params = [p for n, p in model.named_parameters() if "W_E" not in n]
    opt = torch.optim.AdamW(params, lr=lr)
    for step in range(steps):
        x = feature_batch(n_features, batch, feature_prob, device, gen)
        loss = F.mse_loss(model(x), target_fn(x))
        opt.zero_grad(); loss.backward(); opt.step()
    return model


def _self_test() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gen = torch.Generator(device=device).manual_seed(0)

    tms = train_tms(n_features=5, n_hidden=2, steps=5000, batch=1024, feature_prob=0.05,
                    lr=1e-2, device=device, seed=0)
    x = feature_batch(5, 4096, 0.05, device, gen)
    tms_loss = F.mse_loss(tms(x), x).item()
    col_norms = tms.W.weight.detach().norm(dim=0)  # one per feature
    print(f"TMS_5-2: recon mse={tms_loss:.4f} | feature col norms={col_norms.cpu().numpy().round(2)}", flush=True)
    print(f"  ground_truth shape {tuple(tms.ground_truth().shape)} (n_features rank-1 mechanisms)", flush=True)

    rm = train_resid_mlp(n_features=100, d_embed=1000, d_mlp=50, n_layers=2, steps=4000,
                         batch=2048, feature_prob=0.01, lr=3e-3, device=device, seed=0)
    x = feature_batch(100, 4096, 0.01, device, gen)
    rm_loss = F.mse_loss(rm(x), target_fn(x)).item()
    # baseline: predicting just the residual passthrough (no MLP) = embedding readoff of x
    base = F.mse_loss(x @ rm.W_E @ rm.W_E.t(), target_fn(x)).item()
    print(f"ResidMLP 2L (cross-layer): mse={rm_loss:.4f} vs no-MLP baseline {base:.4f}", flush=True)
    print("TOY_MODELS OK", flush=True)


if __name__ == "__main__":
    _self_test()
