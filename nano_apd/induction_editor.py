"""Memory-efficient exact-residual editor for cross-layer induction runs."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from nano_apd.carving import bank_for


class InductionEditor:
    """Patch selected linears with ``W - sum_c (1-g_c) P_c``.

    Unlike the original carving editor, this implementation reduces over components
    in low-rank space and never allocates [batch, tokens, components, d_out].
    That difference makes all-layer C=32 practical on Pythia-410M.
    """

    def __init__(self, target: nn.Module, banks: nn.ModuleDict, module_paths: list[str]):
        self.target = target
        self.banks = banks
        self.module_paths = list(module_paths)
        self.masks: Tensor | None = None
        self.mask_overrides: dict[str, Tensor] = {}
        self.capture = False
        self.cache: dict[str, dict[str, Tensor]] = {}
        self._original: dict[str, object] = {}
        for path in self.module_paths:
            linear = target.get_submodule(path)
            self._original[path] = linear.forward
            linear.forward = self._make_forward(path, linear)

    @staticmethod
    def _weighted_output(
        x: Tensor, bank: nn.Module, target_weight: Tensor, scales: Tensor
    ) -> Tensor:
        A, B = bank.factors(target_weight.detach().t())
        inner = torch.einsum("btd,cdr->btcr", x, A)
        inner = inner * scales.unsqueeze(-1)
        return torch.einsum("btcr,cro->bto", inner, B)

    def _make_forward(self, path: str, linear: nn.Linear):
        def forward(x: Tensor) -> Tensor:
            if x.ndim != 3:
                raise ValueError(f"induction editor expects [batch, token, hidden] at {path}")
            out = F.linear(x, linear.weight, linear.bias)
            mask = self.mask_overrides.get(path, self.masks)
            if mask is not None:
                removal = 1.0 - mask
                out = out - self._weighted_output(
                    x, bank_for(self.banks, path), linear.weight, removal
                )
            if self.capture:
                if not out.requires_grad and torch.is_grad_enabled():
                    out.requires_grad_(True)
                self.cache[path] = {"pre": x, "post": out}
            return out

        return forward

    def start_capture(self) -> None:
        self.capture = True
        self.cache = {}

    def stop_capture(self) -> None:
        self.capture = False

    def set_recovery(self, banks: nn.ModuleDict | None) -> None:
        if banks is not None:
            raise ValueError("InductionEditor does not support recovery banks")

    def restore(self) -> None:
        for path, forward in self._original.items():
            self.target.get_submodule(path).forward = forward
        self._original.clear()
