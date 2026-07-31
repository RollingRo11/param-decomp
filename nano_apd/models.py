"""Toy models for nano_apd: TMS and residual MLP, target + APD versions, plus datasets
and target-model training. Mirrors spd/experiments/{tms,resid_mlp} in the reference repo.

Both experiments keep an explicit n_instances dimension (the reference trains several
independent instances of each toy model in parallel; resid-mlp uses n_instances=1).
"""

from dataclasses import dataclass

import einops
import torch
import torch.nn.functional as F
from torch import Tensor, nn

from nano_apd.apd import (
    Linear,
    LinearComponent,
    TransposedLinearComponent,
    get_lr_schedule_fn,
    init_param_,
)


# ---------------------------------------------------------------------------
# Data (mirrors spd/utils.py:SparseFeatureDataset)
# ---------------------------------------------------------------------------

class SparseFeatureDataset:
    """Each feature is active independently with probability feature_probability; active
    values are uniform in value_range."""

    def __init__(self, n_instances: int, n_features: int, feature_probability: float,
                 device: str, value_range: tuple[float, float]):
        self.n_instances = n_instances
        self.n_features = n_features
        self.feature_probability = feature_probability
        self.device = device
        self.value_range = value_range

    def generate_batch(self, batch_size: int) -> Tensor:  # [batch, n_instances, n_features]
        min_val, max_val = self.value_range
        total = batch_size * self.n_instances
        batch = torch.rand((total, self.n_features), device=self.device) * (max_val - min_val) + min_val
        mask = torch.rand_like(batch) < self.feature_probability
        batch = batch * mask
        return einops.rearrange(batch, "(b i) f -> b i f", b=batch_size)


# ---------------------------------------------------------------------------
# TMS (toy model of superposition)
# ---------------------------------------------------------------------------

@dataclass
class TMSConfig:
    n_instances: int
    n_features: int
    n_hidden: int
    feature_probability: float
    batch_size: int
    steps: int
    seed: int = 0
    lr: float = 5e-3  # reference train_tms.py train() default (config lr is unused there too)


class TMSModel(nn.Module):
    """y = ReLU(W^T W x + b). linear2 is the tied transpose of linear1."""

    def __init__(self, config: TMSConfig):
        super().__init__()
        self.config = config
        self.linear1 = Linear(config.n_features, config.n_hidden, config.n_instances,
                              init_type="xavier_normal")
        self.b_final = nn.Parameter(torch.zeros(config.n_instances, config.n_features))

    def weights(self) -> dict[str, Tensor]:
        w1 = self.linear1.weight
        return {"linear1": w1, "linear2": einops.rearrange(w1, "i f h -> i h f")}

    def forward(self, x: Tensor) -> tuple[Tensor, dict]:
        cache = {}
        hidden = self.linear1(x, cache=cache, name="linear1")
        w2 = einops.rearrange(self.linear1.weight, "i f h -> i h f")
        out_pre_bias = einops.einsum(hidden, w2, "batch i h, i h f -> batch i f")
        cache["linear2"] = {"pre": hidden, "post": out_pre_bias}
        out = F.relu(out_pre_bias + self.b_final)
        return out, cache


class TMSAPDModel(nn.Module):
    def __init__(self, config: TMSConfig, C: int, m: int | None):
        super().__init__()
        self.config = config
        self.C = C
        self.m = min(config.n_features, config.n_hidden) + 1 if m is None else m
        self.linear1 = LinearComponent(config.n_features, config.n_hidden, C=C, m=self.m,
                                       n_instances=config.n_instances, init_type="xavier_normal")
        self.linear2 = TransposedLinearComponent(self.linear1)
        self.b_final = nn.Parameter(torch.zeros(config.n_instances, config.n_features))

    def weights(self) -> dict[str, Tensor]:
        return {"linear1": self.linear1.weight, "linear2": self.linear2.weight}

    def component_weights(self) -> dict[str, Tensor]:
        return {"linear1": self.linear1.component_weights,
                "linear2": self.linear2.component_weights}

    def As(self) -> dict[str, Tensor]:
        return {"linear1": self.linear1.A, "linear2": self.linear2.A}

    def Bs(self) -> dict[str, Tensor]:
        return {"linear1": self.linear1.B, "linear2": self.linear2.B}

    def set_As_to_unit_norm(self) -> None:
        self.linear1.A.data /= self.linear1.A.data.norm(p=2, dim=-2, keepdim=True)

    def fix_normalized_adam_gradients(self) -> None:
        from nano_apd.apd import remove_grad_parallel_to_subnetwork_vecs
        remove_grad_parallel_to_subnetwork_vecs(self.linear1.A.data, self.linear1.A.grad)

    def forward(self, x: Tensor, topk_mask: Tensor | None = None) -> tuple[Tensor, dict]:
        cache = {}
        hidden = self.linear1(x, topk_mask=topk_mask, cache=cache, name="linear1")
        out_pre_bias = self.linear2(hidden, topk_mask=topk_mask, cache=cache, name="linear2")
        out = F.relu(out_pre_bias + self.b_final)
        return out, cache


def train_tms(config: TMSConfig, device: str) -> TMSModel:
    """Reference train_tms.py: AdamW, lr 5e-3 linear decay, loss mean over batch/features,
    summed over instances."""
    torch.manual_seed(config.seed)
    model = TMSModel(config).to(device)
    dataset = SparseFeatureDataset(config.n_instances, config.n_features,
                                   config.feature_probability, device, value_range=(0.0, 1.0))
    opt = torch.optim.AdamW(model.parameters(), lr=config.lr)
    for step in range(config.steps):
        step_lr = config.lr * (1 - step / config.steps)
        for group in opt.param_groups:
            group["lr"] = step_lr
        opt.zero_grad(set_to_none=True)
        batch = dataset.generate_batch(config.batch_size)
        out, _ = model(batch)
        error = (batch.abs() - out) ** 2
        loss = einops.reduce(error, "b i f -> i", "mean").sum()
        loss.backward()
        opt.step()
        if step % 500 == 0 or step + 1 == config.steps:
            print(f"tms target step {step} loss/instance {loss.item() / config.n_instances:.3e}",
                  flush=True)
    return model


# ---------------------------------------------------------------------------
# Residual MLP (toy models of compressed computation [1 layer] and of cross-layer
# distributed representations [2 layers])
# ---------------------------------------------------------------------------

@dataclass
class ResidMLPConfig:
    n_instances: int
    n_features: int
    d_embed: int
    d_mlp: int          # per layer
    n_layers: int
    feature_probability: float
    batch_size: int
    steps: int
    seed: int = 0
    lr: float = 3e-3
    label_fn_seed: int = 0
    in_bias: bool = False  # bias on mlp_in (the repo's TMDR benchmark target has one)


class ResidMLPModel(nn.Module):
    """resid = W_E x; resid += MLP_l(resid) for each layer; y = W_U resid.
    W_E is fixed random with unit-norm rows, W_U = W_E^T (both frozen). ReLU MLPs, no biases.
    Labels for training: y_i = x_i + ReLU(x_i) (trivial label coefficients, as in the
    reference's use_trivial_label_coeffs=True default used for the paper models)."""

    def __init__(self, config: ResidMLPConfig):
        super().__init__()
        self.config = config
        W_E = torch.randn(config.n_instances, config.n_features, config.d_embed)
        W_E = W_E / W_E.norm(dim=-1, keepdim=True)
        self.W_E = nn.Parameter(W_E, requires_grad=False)
        self.W_U = nn.Parameter(einops.rearrange(W_E, "i f e -> i e f").clone(),
                                requires_grad=False)
        self.mlp_in = nn.ModuleList(
            [Linear(config.d_embed, config.d_mlp, config.n_instances, init_type="kaiming_uniform")
             for _ in range(config.n_layers)]
        )
        self.mlp_out = nn.ModuleList(
            [Linear(config.d_mlp, config.d_embed, config.n_instances, init_type="kaiming_uniform")
             for _ in range(config.n_layers)]
        )
        self.bias1 = None
        if config.in_bias:
            self.bias1 = nn.ParameterList(
                [nn.Parameter(torch.zeros(config.n_instances, config.d_mlp))
                 for _ in range(config.n_layers)]
            )

    def param_names(self) -> list[str]:
        names = []
        for i in range(self.config.n_layers):
            names += [f"layers.{i}.mlp_in", f"layers.{i}.mlp_out"]
        return names

    def weights(self) -> dict[str, Tensor]:
        w = {}
        for i in range(self.config.n_layers):
            w[f"layers.{i}.mlp_in"] = self.mlp_in[i].weight
            w[f"layers.{i}.mlp_out"] = self.mlp_out[i].weight
        return w

    def forward(self, x: Tensor, perturb: tuple[int, Tensor] | None = None) -> tuple[Tensor, dict]:
        cache = {}
        residual = einops.einsum(x, self.W_E, "b i f, i f e -> b i e")
        for l in range(self.config.n_layers):
            if perturb is not None and perturb[0] == l:
                residual = residual + perturb[1]
            mid = self.mlp_in[l](residual, cache=cache, name=f"layers.{l}.mlp_in")
            if self.bias1 is not None:
                mid = mid + self.bias1[l]  # cache "post" stays pre-bias, as in the reference
            out = self.mlp_out[l](F.relu(mid), cache=cache, name=f"layers.{l}.mlp_out")
            residual = residual + out
        out = einops.einsum(residual, self.W_U, "b i e, i e f -> b i f")
        return out, cache


class ResidMLPAPDModel(nn.Module):
    def __init__(self, config: ResidMLPConfig, C: int, m: int | None, init_scale: float = 1.0):
        super().__init__()
        self.config = config
        self.C = C
        self.m = min(config.d_embed, config.d_mlp) if m is None else m
        # W_E / W_U are copied from the target and frozen by the run script.
        self.W_E = nn.Parameter(torch.empty(config.n_instances, config.n_features, config.d_embed),
                                requires_grad=False)
        self.W_U = nn.Parameter(torch.empty(config.n_instances, config.d_embed, config.n_features),
                                requires_grad=False)
        self.mlp_in = nn.ModuleList(
            [LinearComponent(config.d_embed, config.d_mlp, C=C, m=self.m,
                             n_instances=config.n_instances, init_type="xavier_normal",
                             init_scale=init_scale)
             for _ in range(config.n_layers)]
        )
        self.mlp_out = nn.ModuleList(
            [LinearComponent(config.d_mlp, config.d_embed, C=C, m=self.m,
                             n_instances=config.n_instances, init_type="xavier_normal",
                             init_scale=init_scale)
             for _ in range(config.n_layers)]
        )
        self.bias1 = None
        if config.in_bias:
            # copied from the target and frozen by the run script
            self.bias1 = nn.ParameterList(
                [nn.Parameter(torch.zeros(config.n_instances, config.d_mlp), requires_grad=False)
                 for _ in range(config.n_layers)]
            )

    def _components(self) -> dict[str, LinearComponent]:
        comps = {}
        for i in range(self.config.n_layers):
            comps[f"layers.{i}.mlp_in"] = self.mlp_in[i]
            comps[f"layers.{i}.mlp_out"] = self.mlp_out[i]
        return comps

    def weights(self) -> dict[str, Tensor]:
        return {n: c.weight for n, c in self._components().items()}

    def component_weights(self) -> dict[str, Tensor]:
        return {n: c.component_weights for n, c in self._components().items()}

    def As(self) -> dict[str, Tensor]:
        return {n: c.A for n, c in self._components().items()}

    def Bs(self) -> dict[str, Tensor]:
        return {n: c.B for n, c in self._components().items()}

    def set_As_to_unit_norm(self) -> None:
        for c in self._components().values():
            c.A.data /= c.A.data.norm(p=2, dim=-2, keepdim=True)

    def fix_normalized_adam_gradients(self) -> None:
        from nano_apd.apd import remove_grad_parallel_to_subnetwork_vecs
        for c in self._components().values():
            remove_grad_parallel_to_subnetwork_vecs(c.A.data, c.A.grad)

    def forward(self, x: Tensor, topk_mask: Tensor | None = None,
                perturb: tuple[int, Tensor] | None = None) -> tuple[Tensor, dict]:
        cache = {}
        residual = einops.einsum(x, self.W_E, "b i f, i f e -> b i e")
        for l in range(self.config.n_layers):
            if perturb is not None and perturb[0] == l:
                residual = residual + perturb[1]
            mid = self.mlp_in[l](residual, topk_mask=topk_mask, cache=cache,
                                 name=f"layers.{l}.mlp_in")
            if self.bias1 is not None:
                mid = mid + self.bias1[l]
            out = self.mlp_out[l](F.relu(mid), topk_mask=topk_mask, cache=cache,
                                  name=f"layers.{l}.mlp_out")
            residual = residual + out
        out = einops.einsum(residual, self.W_U, "b i e, i e f -> b i f")
        return out, cache


def resid_mlp_labels(batch: Tensor) -> Tensor:
    """act_plus_resid labels with trivial (all-ones) coefficients: y = ReLU(x) + x."""
    return F.relu(batch) + batch


def train_resid_mlp(config: ResidMLPConfig, device: str) -> ResidMLPModel:
    """Reference train_resid_mlp.py defaults: AdamW(wd=0.01), lr 3e-3 cosine, MSE to
    y = ReLU(x) + x on the readoff, embedding fixed."""
    torch.manual_seed(config.seed)
    model = ResidMLPModel(config).to(device)
    dataset = SparseFeatureDataset(config.n_instances, config.n_features,
                                   config.feature_probability, device, value_range=(-1.0, 1.0))
    trainable = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(trainable, lr=config.lr, weight_decay=0.01)
    lr_schedule_fn = get_lr_schedule_fn("cosine")
    for step in range(config.steps):
        step_lr = config.lr * lr_schedule_fn(step, config.steps)
        for group in opt.param_groups:
            group["lr"] = step_lr
        opt.zero_grad(set_to_none=True)
        batch = dataset.generate_batch(config.batch_size)
        labels = resid_mlp_labels(batch)
        out, _ = model(batch)
        loss = ((out - labels) ** 2).mean(dim=(0, 2)).mean()
        loss.backward()
        opt.step()
        if step % 500 == 0 or step + 1 == config.steps:
            print(f"resid target step {step} loss {loss.item():.3e}", flush=True)
    return model
