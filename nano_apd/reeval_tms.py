"""Re-evaluate a finished TMS APD run from its saved checkpoint (no retraining).

    python -m nano_apd.reeval_tms --variant 5-2
"""

import argparse
import json

import torch

from nano_apd.models import TMSAPDModel, TMSModel
from nano_apd.run_tms import OUT_ROOT, TMS_VARIANTS, eval_mmcs_ml2r, get_target


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=list(TMS_VARIANTS), default="5-2")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    variant = TMS_VARIANTS[args.variant]
    tms_config, apd_config = variant["tms"], variant["apd"]
    out_dir = OUT_ROOT / f"tms_{args.variant}"

    target = get_target(tms_config, args.device)
    apd_model = TMSAPDModel(tms_config, C=apd_config.C, m=apd_config.m).to(args.device)
    apd_model.load_state_dict(
        torch.load(out_dir / "apd_model.pth", weights_only=True, map_location=args.device)
    )

    summary = eval_mmcs_ml2r(target, apd_model)
    with open(out_dir / "summary_paper_metrics.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps({k: v for k, v in summary.items() if k != "best_component_per_feature"},
                     indent=2))


if __name__ == "__main__":
    main()
