"""
run_glare_parallel.py — GLARE scoring split across 2 GPUs by epoch.

GPU 0 scores epochs [0 .. N//2-1]
GPU 1 scores epochs [N//2 .. N-1]

Results are mathematically identical to the single-GPU run:
  GLARE(x) = sum_t 1[argmin_c ||grad L(x,c)|| == y]
Each term is computed from a fixed checkpoint and is independent of all
other epochs and samples, so any split strategy gives the same final sum.

Runtime: ~half that of run_glare.py.
"""

import gc
import json
import os
import re
import time
from datetime import datetime
from pathlib import Path

import pandas as pd
import psutil
import torch
import torch.multiprocessing as mp

from dataloaders.dataloader_VerSe_new import VerSeDataLoader
from glare import calculate_gradnorm, calculate_glare
from helper.arguments import get_parameter
from models.model_densenet import DenseNetModel
from monai.transforms import Compose, NormalizeIntensity


class ImageTransform:
    def __init__(self, mean, std):
        self.transforms = Compose([
            NormalizeIntensity(subtrahend=mean, divisor=std, channel_wise=True),
        ])

    def __call__(self, data):
        return self.transforms(data)


def glare_worker(gpu_id, epoch_weights, epoch_offset, spec, spec_data, out_path):
    """
    Runs in a separate process.  Scores `epoch_weights` on `cuda:{gpu_id}`.
    Writes a grad_norms DataFrame with globally-correct epoch indices to `out_path`.
    """
    device = torch.device(f"cuda:{gpu_id}")
    print(f"[GPU {gpu_id}] Starting — {len(epoch_weights)} epochs, offset={epoch_offset}", flush=True)

    # Re-create dataloader inside the worker to avoid CUDA fork issues.
    transform = ImageTransform(mean=spec_data["mean"], std=spec_data["std"])
    data_module = VerSeDataLoader(
        spec=spec,
        train_transforms=transform,
        val_transforms=transform,
        num_workers=4,
    )
    data_module.setup()
    train_loader = data_module.train_dataloader()

    weights_dir = str(Path(epoch_weights[0]).parent)

    models = []
    for weight_path in epoch_weights:
        full_model = DenseNetModel(num_classes=4, spec=spec, weights_dir=weights_dir)
        state_dict = torch.load(weight_path, map_location=device)
        full_model.load_state_dict(state_dict)
        net = full_model.model.to(device)
        models.append(net)
        print(f"[GPU {gpu_id}]   loaded {Path(weight_path).name}", flush=True)

    grad_norms = calculate_gradnorm(
        models=models,
        train_dataloader=train_loader,
        device=device,
        num_classes=4,
    )

    # Shift local epoch indices (0, 1, …) to their global positions.
    grad_norms["epoch"] = grad_norms["epoch"] + epoch_offset

    grad_norms.to_pickle(out_path)
    print(f"[GPU {gpu_id}] Done — {len(grad_norms)} rows saved to {out_path}", flush=True)


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("No CUDA GPU found.")
    if torch.cuda.device_count() < 2:
        raise RuntimeError(
            f"Dual-GPU run requires 2 GPUs; only {torch.cuda.device_count()} found. "
            "Use run_glare.py for single-GPU."
        )

    # ── Config ────────────────────────────────────────────────────────────
    spec_path = "/home/student/lisa_ma/results/VerSe_classifier/test_glare_21082026_2132/spec.json"
    with open(spec_path) as f:
        spec = json.load(f)

    spec_data = {"mean": [-111.3885], "std": [406.3665]}
    spec["dataset_dir"] = "/home/student/lisa_ma/prepared"
    spec["batch_size"] = 1

    # Derive weights_dir from spec_path so a single sed update is enough.
    weights_dir = os.path.join(os.path.dirname(spec_path), "weights")

    print(f"spec_path:   {spec_path}")
    print(f"weights_dir: {weights_dir}")
    for i in range(torch.cuda.device_count()):
        print(f"GPU {i}: {torch.cuda.get_device_name(i)}")

    # ── Discover epoch checkpoints ────────────────────────────────────────
    epoch_pattern = re.compile(r"^epoch_(\d+)$")
    epoch_files = sorted(
        [os.path.join(weights_dir, f) for f in os.listdir(weights_dir) if epoch_pattern.match(f)],
        key=lambda x: int(x.rsplit("_", 1)[-1]),
    )
    num_epochs = len(epoch_files)
    if num_epochs == 0:
        raise RuntimeError(f"No epoch checkpoints found in {weights_dir}")

    print(f"\nFound {num_epochs} epoch checkpoints: {[Path(w).name for w in epoch_files]}")

    # ── Split epochs ──────────────────────────────────────────────────────
    split = num_epochs // 2
    gpu0_epochs = epoch_files[:split]
    gpu1_epochs = epoch_files[split:]

    print(f"GPU 0 ({split} epochs):          {[Path(w).name for w in gpu0_epochs]}")
    print(f"GPU 1 ({num_epochs - split} epochs): {[Path(w).name for w in gpu1_epochs]}")

    # ── Output paths ──────────────────────────────────────────────────────
    results_base = os.path.join("results", "glare")
    os.makedirs(results_base, exist_ok=True)

    now = datetime.now()
    run_ts = now.strftime("%Y%m%d_%H%M%S")
    timestamp = now.strftime("%d-%m-%y_%H:%M")

    run_folder = f"{timestamp}_ep-{num_epochs}_dual-gpu"
    run_dir = os.path.join(results_base, run_folder)
    os.makedirs(run_dir, exist_ok=True)

    tmp0 = os.path.join(results_base, f"_tmp_gpu0_{run_ts}.pkl")
    tmp1 = os.path.join(results_base, f"_tmp_gpu1_{run_ts}.pkl")

    # ── Spawn workers ─────────────────────────────────────────────────────
    ctx = mp.get_context("spawn")
    p0 = ctx.Process(target=glare_worker, args=(0, gpu0_epochs, 0,     spec, spec_data, tmp0))
    p1 = ctx.Process(target=glare_worker, args=(1, gpu1_epochs, split, spec, spec_data, tmp1))

    print("\nStarting GPU workers...")
    wall_start = time.time()
    p0.start()
    p1.start()
    p0.join()
    p1.join()
    elapsed = time.time() - wall_start

    if p0.exitcode != 0 or p1.exitcode != 0:
        raise RuntimeError(
            f"Worker failed — GPU 0 exitcode={p0.exitcode}, GPU 1 exitcode={p1.exitcode}"
        )

    # ── Merge grad_norms and compute scores ───────────────────────────────
    print("\nMerging grad_norms from both GPUs...")
    gn0 = pd.read_pickle(tmp0)
    gn1 = pd.read_pickle(tmp1)
    grad_norms = (
        pd.concat([gn0, gn1], ignore_index=True)
        .sort_values(["id", "epoch"])
        .reset_index(drop=True)
    )
    print(f"Merged rows: {len(grad_norms)}  |  Epochs: {sorted(grad_norms['epoch'].unique())}")

    scores = calculate_glare(grad_norms)

    # ── Threshold and report ──────────────────────────────────────────────
    method = get_parameter(spec, "method", "glarex", str)
    threshold = scores[f"{method} score"].quantile(
        get_parameter(spec, "threshold_fraction", 0.1, float)
    )
    mislabels = scores[scores[f"{method} score"] <= threshold]

    print(f"\nAnalysed samples:   {len(scores)}")
    print(f"Identified mislabels: {len(mislabels)} ({100*len(mislabels)/len(scores):.1f}%)")
    print(f"Threshold ({method}): {threshold:.6f}")
    print(f"Wall time: {elapsed/60:.1f} min")

    # ── Save results ──────────────────────────────────────────────────────
    scores.to_csv(os.path.join(run_dir, f"mislabels_list_{run_ts}.csv"), index=False)
    mislabels.to_csv(os.path.join(run_dir, f"mislabels_only_{run_ts}.csv"), index=False)
    scores.to_pickle(os.path.join(run_dir, f"scores_{run_ts}.pkl"))
    grad_norms.to_pickle(os.path.join(run_dir, f"grad_norms_{run_ts}.pkl"))

    run_info = {
        "timestamp": now.isoformat(),
        "run_ts": run_ts,
        "spec_path": spec_path,
        "weights_dir": weights_dir,
        "num_epochs": num_epochs,
        "gpu0_epochs": [Path(w).name for w in gpu0_epochs],
        "gpu1_epochs": [Path(w).name for w in gpu1_epochs],
        "gpu0_name": torch.cuda.get_device_name(0),
        "gpu1_name": torch.cuda.get_device_name(1),
        "processing_time_minutes": round(elapsed / 60, 2),
        "total_samples": len(scores),
        "identified_mislabels": len(mislabels),
    }
    with open(os.path.join(run_dir, "run_info.json"), "w") as f:
        json.dump(run_info, f, indent=2)

    # ── Cleanup temp files ────────────────────────────────────────────────
    os.remove(tmp0)
    os.remove(tmp1)

    print(f"\nResults saved to: {run_dir}")


if __name__ == "__main__":
    main()
