# Raw induction-component reports

These are the compact reports behind
[`INDUCTION_COMPONENT_RESULTS.md`](../../INDUCTION_COMPONENT_RESULTS.md).  The C=8,
C=16, and C=32 directories contain the exact training configuration, routed held-out
evaluation, controlled functional-role audit, and natural-text audit.  C=16 also
contains the final 384-head attribution/direct-ablation report.

The learned `banks.pt` files are intentionally omitted because model artifacts belong in
external artifact storage, not Git.  The paths recorded inside the JSON files refer to
their original local artifact directories.  These runs predate the trainer's individual
natural-text ablation loss; the Markdown results document distinguishes measured results
from that next-run objective change.
