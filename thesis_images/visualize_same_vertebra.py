"""
Standalone script to compare the SAME vertebra level across multiple different
patients/scans, to illustrate scanner noise and other CT imaging challenges
for the thesis theory chapter.

Reuses VerSeDataLoader / VerSeDataset as-is (no changes to existing code).
Run with: python visualize_same_vertebra.py
"""

import os
import json
from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from dataloaders.dataloader_VerSe import VerSeDataLoader, VerSeDataset

TARGET_VERT = 20  # ORIGINAL (uncorrected) vertebra number, as parsed from the filename
NUM_SAMPLES = 4
SAMPLE_OFFSET = 0  # skip this many leading candidates (e.g. 1 = drop the 1st, keep next 4)
OUTPUT_DIR = "thesis_images"

# Manually reviewed ground-truth anomaly/mislabel table (patient-level flags around
# the T12/L1 numbering boundary). Patients flagged here are excluded from selection.
GT_EXCEL = "/home/student/lisa_ma/evaluation/A_CT_ANOMALY_LABELv9.xlsx"

# Maps pid -> source dataset (dsname); used to restrict selection to public VerSe
# patients only (dataset-verse19 / dataset-verse20), excluding other clinical cohorts.
DATA_FILTER_EXCEL = "/home/student/lisa_ma/datasets/VerSe/data_filter_joined.xlsx"
ALLOWED_DSNAMES = {"dataset-verse19", "dataset-verse20"}

spec = {
    "directory": "results/VerSe_classifier",
    "dataset_dir": "/home/student/lisa_ma/prepared",
    "job_name": "viz_same_vert",
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


def get_flagged_mislabel_pids(gt_excel_path):
    """PIDs ('fid') flagged as known mislabels in the manually reviewed GT excel:
    T11==1 or T13==1 (vertebra 19/20 mislabeled per compare_full.py convention),
    a non-empty LabelOverride, or Remove==1."""
    if not os.path.isfile(gt_excel_path):
        print(f"Warning: GT excel not found at {gt_excel_path}, skipping mislabel filter.")
        return set()

    df_gt = pd.read_excel(gt_excel_path, dtype=str, keep_default_na=False)
    flagged = df_gt[
        (df_gt.get("T11") == "1")
        | (df_gt.get("T13") == "1")
        | (df_gt.get("LabelOverride", "").str.strip() != "")
        | (df_gt.get("Remove") == "1")
    ]
    return set(flagged["fid"].astype(str).str.strip())


def get_verse_pids(data_filter_excel_path, allowed_dsnames):
    """PIDs whose dsname is in allowed_dsnames (i.e. genuinely from the public
    VerSe dataset, not one of the other clinical source cohorts)."""
    df = pd.read_excel(data_filter_excel_path)
    verse = df[df["dsname"].isin(allowed_dsnames)]
    return set(verse["pid"].astype(str).str.strip())


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

    dataset = data_module.train_dataset

    flagged_pids = get_flagged_mislabel_pids(GT_EXCEL)
    print(f"Excluding {len(flagged_pids)} patients flagged as known mislabels in {GT_EXCEL}")

    verse_pids = get_verse_pids(DATA_FILTER_EXCEL, ALLOWED_DSNAMES)
    print(f"Restricting selection to {len(verse_pids)} patients from {sorted(ALLOWED_DSNAMES)}")

    # Group one sample index per unique patient, keyed by the ORIGINAL vertebra
    # number parsed directly from the filename (no pid_corrections applied).
    vert_to_pid_idx = defaultdict(dict)  # vert -> {pid: idx}
    skipped_flagged = 0
    skipped_non_verse = 0
    for idx, path in enumerate(dataset.file_paths):
        orig_vert = VerSeDataset.get_vert_number(path)
        pid = VerSeDataset.get_pid_from_path(path)
        if pid in flagged_pids:
            skipped_flagged += 1
            continue
        if pid not in verse_pids:
            skipped_non_verse += 1
            continue
        vert_to_pid_idx[orig_vert].setdefault(pid, idx)
    print(f"Skipped {skipped_flagged} file(s) belonging to flagged patients.")
    print(f"Skipped {skipped_non_verse} file(s) not belonging to a VerSe patient.")

    diversity_ranked = sorted(
        vert_to_pid_idx.items(), key=lambda kv: -len(kv[1])
    )
    print("Vertebra levels with the most distinct (unflagged) patients available:")
    for vert, pid_idx in diversity_ranked[:10]:
        print(f"  vert{vert}: {len(pid_idx)} patients")

    available = len(vert_to_pid_idx.get(TARGET_VERT, {}))
    if available < SAMPLE_OFFSET + NUM_SAMPLES:
        raise ValueError(
            f"vert{TARGET_VERT} only has {available} distinct unflagged patients "
            f"(< {SAMPLE_OFFSET + NUM_SAMPLES} needed for offset {SAMPLE_OFFSET} + "
            f"{NUM_SAMPLES} samples). Pick a different TARGET_VERT from the list "
            f"printed above, or lower SAMPLE_OFFSET."
        )

    chosen = list(vert_to_pid_idx[TARGET_VERT].items())[
        SAMPLE_OFFSET : SAMPLE_OFFSET + NUM_SAMPLES
    ]  # [(pid, idx), ...]

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    plt.rcParams.update({"font.family": "serif", "font.size": 10})

    fig, axes = plt.subplots(1, NUM_SAMPLES, figsize=(3 * NUM_SAMPLES, 3.5))
    saved_paths = []

    for i, (pid, idx) in enumerate(chosen):
        image_tensor, label, sample_id = dataset[idx]
        display_slice = get_display_slice(image_tensor)

        individual_path = os.path.join(
            OUTPUT_DIR, f"vert{TARGET_VERT}_sample_{i:02d}_{sample_id}.png"
        )
        plt.imsave(individual_path, display_slice, cmap="gray")
        saved_paths.append(individual_path)

        ax = axes[i]
        ax.imshow(display_slice, cmap="gray")
        ax.set_title(f"Scan {i + 1}", fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])

    plt.tight_layout()
    grid_path = os.path.join(OUTPUT_DIR, f"vert{TARGET_VERT}_comparison.png")
    fig.savefig(grid_path, dpi=300, bbox_inches="tight")

    print(f"\nSaved comparison figure: {grid_path}")
    print(f"Saved {len(saved_paths)} individual slices to '{OUTPUT_DIR}/':")
    for p in saved_paths:
        print(f"  {p}")


if __name__ == "__main__":
    main()
