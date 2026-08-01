# Checked-in token-carving results

These are compact copies of the corrected Pythia-410M rank-4 paired-projection runs:

- [`euclidean/results_compact.json`](euclidean/results_compact.json)
- [`diag_kfac/results_compact.json`](diag_kfac/results_compact.json)

They retain configurations, aggregate metrics, candidate records, confidence intervals,
and interpretability diagnostics while removing large per-example arrays. The complete
raw `results.json`, `config.json`, and reloadable `pieces.pt` artifacts remain in
`nano_apd/out/`, which is intentionally ignored by git. See
[`../../TOKEN_WEIGHT_CARVING_RESULTS.md`](../../TOKEN_WEIGHT_CARVING_RESULTS.md) for the
protocol, results, limitations, and reproduction commands.
