"""Retrain a crisper 2-layer residual MLP target: more neurons + longer training so it actually fits
y_i = x_i + ReLU(x_i) (the cached one undershot the nonlinear part by ~20%). Backs up the old file,
saves the new state_dict in place, then re-probes separability."""

import shutil

import torch
import torch.nn.functional as F

from .resid_leakage_probe import load_cached, leakage_matrix
from .toy_models import ResidMLP, feature_batch, target_fn

PATH = "/tmp/toy/resid_2l.pt"
N_FEATURES = 100
D_EMBED = 256
D_MLP = 60          # 30 per layer (was 20) — more capacity for the ReLU part
N_LAYERS = 2
STEPS = 40000
BATCH = 4096
FPROB = 0.01
LR = 3e-3


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ResidMLP(N_FEATURES, D_EMBED, D_MLP, N_LAYERS, seed=0).to(device)
    gen = torch.Generator(device=device).manual_seed(1)
    params = [p for n, p in model.named_parameters() if "W_E" not in n]
    opt = torch.optim.AdamW(params, lr=LR)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, STEPS, eta_min=LR * 0.05)

    for step in range(STEPS):
        x = feature_batch(N_FEATURES, BATCH, FPROB, device, gen)
        loss = F.mse_loss(model(x), target_fn(x))
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        if step % 5000 == 0 or step == STEPS - 1:
            print(f"step {step:6d}  mse={loss.item():.6f}", flush=True)

    shutil.copyfile(PATH, PATH.replace(".pt", "_v1_undertrained.pt"))
    torch.save(model.state_dict(), PATH)
    print(f"saved new target to {PATH} (backed up old -> resid_2l_v1_undertrained.pt)", flush=True)

    # re-probe separability + fidelity at typical value
    m = load_cached(PATH, device)
    for v in [0.5, 1.0]:
        L = leakage_matrix(m, v, device); diag = L.diag(); off = (L - torch.diag(diag)).abs()
        tgt = v + max(v, 0.0)
        clean = ((diag - tgt).abs() < 0.1 * tgt).logical_and(off.max(1).values < 0.1 * tgt).sum().item()
        print(f"value={v}: target={tgt:.2f} diag_mean={diag.mean():.3f} "
              f"diag_err={(diag - tgt).abs().mean():.4f} leak_mean={off.mean():.4f} clean={clean}/{N_FEATURES}",
              flush=True)
    print("RETRAIN_DONE", flush=True)


if __name__ == "__main__":
    main()
