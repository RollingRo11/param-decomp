"""Re-probe a SAVED decomposition for induction without re-training. Loads vpd_s*.pt / mpd_s*.pt,
measures copy accuracy on a RANDOM-repeat probe and an IN-DISTRIBUTION-repeat probe (repeat real
model-sampled sequences), under the original model (passthrough) and the CI-masked decomposition.

Tests (a) whether the induction circuit survives CI-masking and (b) whether the earlier ~0
ci-masked result was an OOD-random-probe artifact (in-dist should be higher if so).

    CUDA_VISIBLE_DEVICES="" python -m nano_param_decomp.reprobe_induction /tmp/pythia_compare/vpd_s100000.pt
"""

import os
import sys

os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"

import torch

from . import matryoshka, run
from .run import (
    _require,
    clear_wrapper_masks,
    induction_copy_acc,
    install_components,
    make_induction_batch,
    set_wrapper_masks,
)
from .pythia14m import load_pythia14m_target

CKPT = sys.argv[1] if len(sys.argv) > 1 else "/tmp/pythia_compare/mpd_s100000.pt"
POOL = "/tmp/pythia_compare/pool.pt"
L = 64


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ck = torch.load(CKPT, weights_only=True)
    method = ck["method"]
    cfgd = dict(ck["cfg"])

    target = load_pythia14m_target().to(device).float()
    wrappers = install_components(target, ck["C_per_module"])
    for name, w in wrappers.items():
        w.V.data = ck["wrappers"][name]["V"].to(device)
        w.U.data = ck["wrappers"][name]["U"].to(device)
    module_order = sorted(wrappers.keys())
    d_in = {n: int(w.W_target.shape[1]) for n, w in wrappers.items()}

    if method == "matryoshka":
        cfg = matryoshka.Config(**cfgd)
        ci_fn = matryoshka.MatryoshkaCI(d_in, cfg).to(device)
        ci_fn.load_state_dict(ck["ci_fn"])
        assign = matryoshka.ComponentAssignment(cfg.C_per_module, module_order, cfg).to(device)
        assign.M_logits.data = ck["M_logits"].to(device)
        tau = cfg.tau_end
    else:
        cfg = run.Config(**cfgd)
        ci_fn = run.CITransformer(d_in, cfg.C_per_module, cfg).to(device)
        ci_fn.load_state_dict(ck["ci_fn"])
        assign = None
        tau = None
    ci_fn.eval()

    vocab = target.config.vocab_size
    pool = torch.load(POOL, weights_only=True)
    eval_pool = pool[int(0.9 * pool.shape[0]):]
    indist_first = eval_pool[:32, :L].to(device)
    probes = {
        "random": make_induction_batch(vocab, 32, L, device, seed=0),
        "indist": (torch.cat([indist_first, indist_first], dim=1), indist_first),
    }

    @torch.no_grad()
    def measure(seq: torch.Tensor, first: torch.Tensor) -> tuple[float, float]:
        clear_wrapper_masks(wrappers)
        orig = target(seq)
        acts = {n: _require(w.last_input) for n, w in wrappers.items()}
        ci_low, _u, _p = ci_fn(acts)
        masks = assign.atom_masks(ci_low, tau) if method == "matryoshka" else ci_low
        B, S = seq.shape
        set_wrapper_masks(wrappers, masks, {n: torch.zeros(B, S, device=device) for n in wrappers}, None)
        try:
            cim = target(seq)
        finally:
            clear_wrapper_masks(wrappers)
        return induction_copy_acc(orig, first, L), induction_copy_acc(cim, first, L)

    print(f"=== {method}  {os.path.basename(CKPT)} ===", flush=True)
    for name, (seq, first) in probes.items():
        o, c = measure(seq, first)
        print(f"[{name:7}] original copy={o:.3f}   ci-masked copy={c:.3f}", flush=True)
    print("REPROBE DONE", flush=True)


if __name__ == "__main__":
    main()
