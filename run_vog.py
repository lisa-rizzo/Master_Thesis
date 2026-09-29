"""
run_vog.py — Post-hoc VOG scoring on saved epoch checkpoints.

Loads the same training run used for GLARE (same spec, weights_dir, dataloader)
and computes VOG scores using the last t=5 epoch checkpoints.

Output: results/vog/<timestamp>/vog_scores_<ts>.csv
"""

import json
import os
import time
from datetime import datetime
from pathlib import Path

import torch
from monai.transforms import Compose, NormalizeIntensity

from dataloaders.dataloader_VerSe import VerSeDataLoader
from vog import compute_vog


class ImageTransform:
    def __init__(self, mean, std):
        self.transforms = Compose(
            [NormalizeIntensity(subtrahend=mean, divisor=std, channel_wise=True)]
        )

    def __call__(self, data):
        return self.transforms(data)


def main():
    # ── Config (mirror run_glare_parallel.py) ─────────────────────────────
    spec_path = "/home/student/lisa_ma/results/classifier/test_glare_20072026_0111/spec.json"
    with open(spec_path) as f:
        spec = json.load(f)

    spec_data = {"mean": [-111.3885], "std": [406.3665]}
    spec["dataset_dir"] = "/home/student/lisa_ma/prepared"
    spec["batch_size"] = 1

    weights_dir = os.path.join(os.path.dirname(spec_path), "weights")

    # t = number of most-recent checkpoints to use for VOG variance
    t = 5

    print(f"spec_path:   {spec_path}")
    print(f"weights_dir: {weights_dir}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"t (window):  {t}")

    # ── Dataloader ────────────────────────────────────────────────────────
    transform = ImageTransform(mean=spec_data["mean"], std=spec_data["std"])
    data_module = VerSeDataLoader(
        spec=spec,
        train_transforms=transform,
        val_transforms=transform,
        num_workers=4,
    )
    data_module.setup()
    train_loader = data_module.train_dataloader()
    print(f"Training samples: {len(train_loader.dataset)}")

    # ── Run VOG ───────────────────────────────────────────────────────────
    wall_start = time.time()
    scores = compute_vog(
        dataloader=train_loader,
        weights_dir=weights_dir,
        spec=spec,
        num_classes=4,
        t=t,
    )
    elapsed = time.time() - wall_start

    print(f"\nScored {len(scores)} samples in {elapsed/60:.1f} min")
    print(scores["vog_score"].describe().to_string())

    # ── Save results ──────────────────────────────────────────────────────
    now = datetime.now()
    run_ts = now.strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join("results", "vog", run_ts)
    os.makedirs(run_dir, exist_ok=True)

    csv_path = os.path.join(run_dir, f"vog_scores_{run_ts}.csv")
    scores.to_csv(csv_path, index=False)

    run_info = {
        "timestamp": now.isoformat(),
        "run_ts": run_ts,
        "spec_path": spec_path,
        "weights_dir": weights_dir,
        "t_window": t,
        "total_samples": len(scores),
        "processing_time_minutes": round(elapsed / 60, 2),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "vog_score_mean": round(float(scores["vog_score"].mean()), 6),
        "vog_score_std":  round(float(scores["vog_score"].std()), 6),
    }
    with open(os.path.join(run_dir, "run_info_vog.json"), "w") as f:
        json.dump(run_info, f, indent=2)

    print(f"\nResults saved to: {run_dir}")
    print(f"CSV: {csv_path}")


if __name__ == "__main__":
    main()
