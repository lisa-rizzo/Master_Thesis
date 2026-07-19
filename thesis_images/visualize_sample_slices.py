"""
Standalone script to load 10 sample CT volumes from the VerSe dataloader and
save exemplary mid-axial slice images for the thesis theory chapter.

Reuses VerSeDataLoader / VerSeDataset as-is (no changes to existing code).
Run with: python visualize_sample_slices.py
"""

import os
import json

import numpy as np
import matplotlib.pyplot as plt

from dataloaders.dataloader_VerSe import VerSeDataLoader

NUM_SAMPLES = 10
OUTPUT_DIR = "thesis_images"

REGION_NAMES = {
    0: "cervical",
    1: "thoracic",
    2: "lumbar",
    3: "sacral",
    -1: "unknown",
}

spec = {
    "directory": "results/VerSe_classifier",
    "dataset_dir": "/home/student/lisa_ma/prepared",
    "job_name": "viz_sample",
    "batch_size": 4,
    "holdout_set_size": 0,
    "use_train_for_val": False,
    "train_set_size": 0.8,
}

pid_json_path = "/home/student/lisa_ma/pid_corrections.json"
if os.path.isfile(pid_json_path):
    try:
        with open(pid_json_path, "r") as f:
            pid_data = json.load(f)
        if pid_data.get("pid_fix_1820"):
            spec["pid_fix_1820"] = pid_data["pid_fix_1820"]
        if pid_data.get("pid_fix_28"):
            spec["pid_fix_28"] = pid_data["pid_fix_28"]
    except Exception as e:
        print("Warning: could not load pid_corrections.json:", e)


def get_display_slice(image_tensor):
    """Mid-axial slice of a (1, D, H, W) tensor, min-max normalized to [0, 1]."""
    arr = image_tensor.numpy()[0]  # (D, H, W)
    z = arr.shape[0] // 2
    sl = arr[z]
    mn, mx = sl.min(), sl.max()
    return (sl - mn) / (mx - mn) if mx > mn else np.zeros_like(sl)


def main():
    data_module = VerSeDataLoader(
        spec=spec,
        train_transforms=None,
        val_transforms=None,
        num_workers=0,
    )
    data_module.setup()

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    plt.rcParams.update({"font.family": "serif", "font.size": 10})

    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    saved_paths = []

    for i in range(NUM_SAMPLES):
        image_tensor, label, sample_id = data_module.train_dataset[i]
        display_slice = get_display_slice(image_tensor)
        region = REGION_NAMES.get(label, "unknown")

        individual_path = os.path.join(
            OUTPUT_DIR, f"sample_slice_{i:02d}_{sample_id}.png"
        )
        plt.imsave(individual_path, display_slice, cmap="gray")
        saved_paths.append(individual_path)

        ax = axes[i // 5, i % 5]
        ax.imshow(display_slice, cmap="gray")
        ax.set_title(f"{sample_id}\n({region})", fontsize=8)
        ax.set_xticks([])
        ax.set_yticks([])

    plt.tight_layout()
    grid_path = os.path.join(OUTPUT_DIR, "sample_grid.png")
    fig.savefig(grid_path, dpi=300, bbox_inches="tight")

    print(f"Saved grid figure: {grid_path}")
    print(f"Saved {len(saved_paths)} individual slices to '{OUTPUT_DIR}/':")
    for p in saved_paths:
        print(f"  {p}")


if __name__ == "__main__":
    main()
