"""Shared target-model and module-discovery helpers for carving experiments."""

from __future__ import annotations

import types
from collections.abc import Iterator

import torch.nn as nn
from torch import Tensor

DEFAULT_HF_MODEL = "EleutherAI/pythia-410m"


def load_carving_target(target_kind: str, model_name: str = DEFAULT_HF_MODEL) -> nn.Module:
    """Load a target whose public ``forward(input_ids)`` returns bare logits."""
    if target_kind == "pile4l":
        from nano_param_decomp.pile_4L import load_paper_target_model

        return load_paper_target_model().eval()
    if target_kind != "hf":
        raise ValueError(f"unknown target kind {target_kind!r}")

    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(model_name)
    original_forward = model.forward

    def forward_logits_only(_self: nn.Module, input_ids: Tensor) -> Tensor:
        return original_forward(input_ids=input_ids, use_cache=False).logits

    model.forward = types.MethodType(forward_logits_only, model)
    model.config.use_cache = False
    return model.eval()


def candidate_linear_paths(target: nn.Module, target_kind: str) -> list[str]:
    """Return editable transformer linears, excluding vocabulary/output heads."""
    if target_kind == "pile4l":
        from nano_param_decomp.pile_4L import C_PER_MODULE_4L

        return list(C_PER_MODULE_4L)

    paths = [
        name
        for name, module in target.named_modules()
        if isinstance(module, nn.Linear)
        and name
        and name not in {"lm_head", "embed_out"}
        and (".layers." in name or ".h." in name or ".blocks." in name)
    ]
    if not paths:
        raise ValueError(
            "no transformer nn.Linear modules found; pass a supported causal LM or add "
            "its block naming convention to candidate_linear_paths"
        )
    return paths


def vocab_size(target: nn.Module) -> int:
    config = getattr(target, "config", getattr(target, "cfg", None))
    if config is None or not hasattr(config, "vocab_size"):
        raise AttributeError("could not infer target vocabulary size")
    return int(config.vocab_size)


def one_loader_batch(loader: Iterator[Tensor]) -> Tensor:
    """Take one streaming batch and close the generator, including on failure."""
    try:
        return next(loader)
    finally:
        close = getattr(loader, "close", None)
        if close is not None:
            close()
