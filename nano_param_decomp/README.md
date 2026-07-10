# nano_param_decomp

Whole-network parameter decomposition with a trained mask ("APD basis, S/VPD training").
**Read `method.md`** — the full method, evaluation battery, validation record, and open problems.

## Layout

Core method:
- `apd_mask.py` — component banks, config, losses, toy entry points (TMS, cross-layer resid-MLP)
- `apd_lm.py` — LM training loop (DDP-capable, wandb, best-checkpoint); attn-only-2l entry
- `apd_pythia.py` — Pythia-14M entry (per-module ratio map, module fingerprints)
- `apd_alg.py` — AlgZoo RNN targets (per-timestep gates, hidden-trajectory recon)

Baselines & targets:
- `run.py` — the VPD engine (rank-1 per-matrix atoms, the published baseline we compare against)
- `toy_decompose.py` — VPD toy harness + `mmcs` (MPD mode archived, lazy import)
- `compare_pythia14m.py` — VPD baseline runner on Pythia-14M
- `toy_models.py`, `attn_only_2l.py`, `pythia14m.py`, `pile_4L.py` — targets
- `vpd_cluster.py` — VPD + MDL-clustering comparison on the cross-layer toy

Diagnostics & interpretation:
- `interact_pythia.py` / `interact_apd.py` / `vpd_interact.py` — component interaction probes
- `alg_interaction.py` — pairwise causal-interaction matrix
- `auto_interp.py` — LLM-judged component labeling + detection validation (needs
  `ANTHROPIC_API_KEY` in repo-root `.env`)
- `bound_interaction_probe.py` — interaction-vs-coactivation on the BoundPairs toy
- `profile_apd.py` — per-step cost profiling

Archived:
- `matryoshka/` — the Matryoshka (MPD) line, superseded; see its README
- `attic/` — retired scripts (old sweeps, induction probes, benches); delete when confident

Key checkpoints are mirrored in `out/checkpoints/` (don't trust `/tmp` to persist).
