"""Numerical cross-checks of nano_apd against the reference implementation
(/workspace/apd-reference, package `spd`). Run:

    python -m nano_apd.tests
"""

import sys

import einops
import torch
import torch.nn.functional as F

sys.path.insert(0, "/workspace/apd-reference")

from spd.run_spd import calc_schatten_loss as ref_calc_schatten_loss  # noqa: E402
from spd.utils import calc_grad_attributions as ref_calc_grad_attributions  # noqa: E402
from spd.utils import calc_topk_mask as ref_calc_topk_mask  # noqa: E402

from nano_apd.apd import (  # noqa: E402
    calc_grad_attributions,
    calc_schatten_loss,
    calc_topk_mask,
)
from nano_apd.models import TMSAPDModel, TMSConfig, TMSModel  # noqa: E402

torch.manual_seed(0)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def make_two_layer_graph(batch=7, n_inst=3, d_in=5, d_mid=4, d_out=5, C=6):
    """A small 2-layer graph (relu in between) with explicit pre/post activations."""
    x = torch.randn(batch, n_inst, d_in, device=DEVICE)
    W1 = torch.randn(n_inst, d_in, d_mid, device=DEVICE, requires_grad=True)
    W2 = torch.randn(n_inst, d_mid, d_out, device=DEVICE, requires_grad=True)
    pre1 = x
    post1 = einops.einsum(pre1, W1, "b i a, i a c -> b i c")
    pre2 = F.relu(post1)
    post2 = einops.einsum(pre2, W2, "b i a, i a c -> b i c")
    out = post2 * 2.0 + 1.0
    cw1 = torch.randn(n_inst, C, d_in, d_mid, device=DEVICE)
    cw2 = torch.randn(n_inst, C, d_mid, d_out, device=DEVICE)
    pre = {"l1": pre1, "l2": pre2}
    post = {"l1": post1, "l2": post2}
    cw = {"l1": cw1, "l2": cw2}
    return out, pre, post, cw, C


def test_grad_attributions_vs_reference():
    out, pre, post, cw, C = make_two_layer_graph()
    ours = calc_grad_attributions(out, pre, post, cw, C, vectorized=False)
    ours_vec = calc_grad_attributions(out, pre, post, cw, C, vectorized=True)
    ref = ref_calc_grad_attributions(
        target_out=out,
        pre_weight_acts={k + ".hook_pre": v for k, v in pre.items()},
        post_weight_acts={k + ".hook_post": v for k, v in post.items()},
        component_weights=cw,
        C=C,
    )
    assert torch.allclose(ours, ref, atol=1e-5), (ours - ref).abs().max()
    assert torch.allclose(ours_vec, ref, atol=1e-5), (ours_vec - ref).abs().max()
    print(f"grad attributions match reference (max|diff| loop {(ours - ref).abs().max():.2e}, "
          f"vectorized {(ours_vec - ref).abs().max():.2e})")


def test_topk_mask_vs_reference():
    for batch_topk in [True, False]:
        scores = torch.rand(16, 3, 10, device=DEVICE)
        topk = 2.0
        ours = calc_topk_mask(scores, topk, batch_topk=batch_topk)
        ref = ref_calc_topk_mask(scores, topk, batch_topk=batch_topk)
        assert torch.equal(ours, ref), f"batch_topk={batch_topk}"
    print("topk mask matches reference (batch_topk true/false)")


def test_schatten_vs_reference():
    n_inst, C, d_in, m, d_out = 3, 6, 5, 4, 7
    As = {"l1": torch.randn(n_inst, C, d_in, m, device=DEVICE)}
    Bs = {"l1": torch.randn(n_inst, C, m, d_out, device=DEVICE)}
    mask = torch.rand(16, n_inst, C, device=DEVICE) > 0.5
    n_params = d_in * d_out
    ours = calc_schatten_loss(As, Bs, mask, p=0.9, n_params=n_params)
    ref = ref_calc_schatten_loss(As, Bs, mask, p=0.9, n_params=n_params, device=DEVICE)
    assert torch.allclose(ours, ref, atol=1e-6), (ours - ref).abs().max()
    print(f"schatten loss matches reference (max|diff| {(ours - ref).abs().max():.2e})")


def test_tms_model_consistency():
    cfg = TMSConfig(n_instances=2, n_features=5, n_hidden=2, feature_probability=0.05,
                    batch_size=8, steps=1, seed=0)
    apd = TMSAPDModel(cfg, C=5, m=None).to(DEVICE)
    # summed component weights == weight
    cw = apd.linear1.component_weights.sum(dim=1)
    assert torch.allclose(cw, apd.linear1.weight, atol=1e-6)
    # transposed component is the transpose
    assert torch.allclose(apd.linear2.weight,
                          einops.rearrange(apd.linear1.weight, "i a b -> i b a"), atol=1e-6)
    # all-ones topk mask == no mask
    x = torch.rand(8, 2, 5, device=DEVICE)
    out_none, _ = apd(x)
    out_ones, _ = apd(x, topk_mask=torch.ones(8, 2, 5, dtype=torch.bool, device=DEVICE))
    assert torch.allclose(out_none, out_ones, atol=1e-6)
    # target model forward runs and matches manual computation
    target = TMSModel(cfg).to(DEVICE)
    out, cache = target(x)
    W = target.linear1.weight
    manual = F.relu(einops.einsum(einops.einsum(x, W, "b i f, i f h -> b i h"),
                                  W, "b i h, i f h -> b i f") + target.b_final)
    assert torch.allclose(out, manual, atol=1e-5)
    print("TMS model consistency checks pass")


def test_attribution_vectorized_on_tms():
    cfg = TMSConfig(n_instances=2, n_features=5, n_hidden=2, feature_probability=0.5,
                    batch_size=16, steps=1, seed=0)
    target = TMSModel(cfg).to(DEVICE)
    apd = TMSAPDModel(cfg, C=5, m=None).to(DEVICE)
    x = torch.rand(16, 2, 5, device=DEVICE)
    out, cache = target(x)
    names = ["linear1", "linear2"]
    kwargs = dict(
        target_out=out,
        pre_weight_acts={n: cache[n]["pre"] for n in names},
        post_weight_acts={n: cache[n]["post"] for n in names},
        component_weights=apd.component_weights(),
        C=5,
    )
    a_loop = calc_grad_attributions(**kwargs, vectorized=False)
    a_vec = calc_grad_attributions(**kwargs, vectorized=True)
    assert torch.allclose(a_loop, a_vec, atol=1e-5), (a_loop - a_vec).abs().max()
    print(f"TMS attribution loop==vectorized (max|diff| {(a_loop - a_vec).abs().max():.2e})")


def _tmdr_models():
    from nano_apd.models import ResidMLPAPDModel, ResidMLPConfig, ResidMLPModel
    cfg = ResidMLPConfig(n_instances=1, n_features=12, d_embed=16, d_mlp=6, n_layers=2,
                         feature_probability=0.1, batch_size=8, steps=0, in_bias=True)
    target = ResidMLPModel(cfg).to(DEVICE)
    for l in range(2):
        target.bias1[l].data.normal_(0, 0.1)
    apd = ResidMLPAPDModel(cfg, C=10, m=None).to(DEVICE)
    apd.W_E.data[:] = target.W_E.data
    apd.W_U.data[:] = target.W_U.data
    for l in range(2):
        apd.bias1[l].data[:] = target.bias1[l].data
    x = torch.rand(8, 1, 12, device=DEVICE) * (torch.rand(8, 1, 12, device=DEVICE) < 0.3)
    return target, apd, x


def test_relp_conservation_and_shapes():
    from nano_apd.attributions import calc_relp_attributions, calc_relp_coefficients
    target, apd, x = _tmdr_models()
    out, cache = target(x)
    # conservation at the readout: relevance at resid_final sums to y_o
    import einops as E
    pre_in = cache["layers.1.mlp_in"]["pre"].detach()
    m_out = cache["layers.1.mlp_out"]["post"].detach()
    resid_final = pre_in + m_out
    R_final = E.einsum(target.W_U.detach(), resid_final, "i e f, b i e -> f b i e")
    assert torch.allclose(R_final.sum(dim=-1), E.rearrange(out.detach(), "b i f -> f b i"),
                          atol=1e-4)
    rho = calc_relp_coefficients(target, cache)
    assert set(rho) == set(target.param_names())
    attr = calc_relp_attributions(target, cache, apd.component_weights(), C=10)
    assert attr.shape == (8, 1, 10) and torch.isfinite(attr).all() and (attr >= 0).all()
    print("relp: readout conservation + coefficient shapes/finiteness pass")


def test_ig_attributions():
    from nano_apd.attributions import calc_ig_mask_attributions
    _, apd, x = _tmdr_models()
    attr = calc_ig_mask_attributions(apd, x, C=10, n_alpha=4)
    assert attr.shape == (8, 1, 10) and torch.isfinite(attr).all() and (attr >= 0).all()
    print("ig-over-mask attributions: shape/finiteness pass")


def test_prefix_mask():
    from nano_apd.greedy import _mask_from_prefix_len
    order = torch.tensor([[[3, 0, 2, 1]]], device=DEVICE)
    k = torch.tensor([[2]], device=DEVICE)
    mask = _mask_from_prefix_len(order, k, 4)
    # removing the first k=2 of order (components 3 and 0) leaves {1, 2}
    assert mask[0, 0].tolist() == [0.0, 1.0, 1.0, 0.0]
    print("prefix-removal mask construction correct")


def test_greedy_invariants():
    from nano_apd.greedy import greedy_add_mask, greedy_prune_mask

    def sample_mse(out, tgt):
        return ((out - tgt) ** 2).mean(dim=-1)

    target, apd, x = _tmdr_models()
    target_out, _ = target(x)
    target_out = target_out.detach()

    # huge eps: add selects nothing; prune removes everything
    m = greedy_add_mask(apd, x, target_out, C=10, eps=1e9)
    assert m.sum() == 0
    m = greedy_prune_mask(apd, x, target_out, torch.rand(8, 1, 10, device=DEVICE), eps=1e9)
    assert m.sum() == 0
    # tiny eps with a random (unfaithful) apd model: prune must fall back to all-on
    m = greedy_prune_mask(apd, x, target_out, torch.rand(8, 1, 10, device=DEVICE), eps=1e-12)
    assert (m == 1).all()
    # moderate eps: certified property — per-sample error < eps OR mask is all-on
    for eps in (1e-2, 1e-3):
        m = greedy_prune_mask(apd, x, target_out, torch.rand(8, 1, 10, device=DEVICE), eps=eps)
        out, _ = apd(x, topk_mask=m)
        ok = (sample_mse(out, target_out) < eps) | (m.sum(dim=-1) == m.shape[-1])
        assert ok.all(), (sample_mse(out, target_out), m.sum(-1))
        # greedy_add terminates and marks done samples correctly
        ma = greedy_add_mask(apd, x, target_out, C=10, eps=eps)
        out_a, _ = apd(x, topk_mask=ma)
        done = sample_mse(out_a, target_out) < eps
        full = ma.sum(dim=-1) == ma.shape[-1]
        assert (done | full).all()
    print("greedy add/prune invariants pass (certified error or saturated mask)")


if __name__ == "__main__":
    test_grad_attributions_vs_reference()
    test_topk_mask_vs_reference()
    test_schatten_vs_reference()
    test_tms_model_consistency()
    test_attribution_vectorized_on_tms()
    test_relp_conservation_and_shapes()
    test_ig_attributions()
    test_prefix_mask()
    test_greedy_invariants()
    print("all checks passed")
