"""
Standalone script to build a CIFAR-10-style example grid for the thesis: one row
per vertebral region class (cervical, thoracic, lumbar, sacral) with 2 example
mid-axial slices each, with the class name labelled on the left of each row.

Only patients from the public VerSe subsets (dataset-verse19 / dataset-verse20)
are used. Reuses VerSeDataLoader / VerSeDataset as-is (no changes to existing code).
Run with: python visualize_region_classes.py
"""

import os
import json
from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from dataloaders.dataloader_VerSe import VerSeDataLoader, VerSeDataset

EXAMPLES_PER_CLASS = 4
OUTPUT_DIR = "thesis_images"

# Region label -> display name (matches VerSeDataset.get_region_label_from_vert).
REGION_NAMES = {
    0: "cervical",
    1: "thoracic",
    2: "lumbar",
    3: "sacral",
}
REGION_ORDER = [0, 1, 2, 3]

# Maps pid -> source dataset (dsname); used to restrict selection to public VerSe
# patients only (dataset-verse19 / dataset-verse20), excluding other clinical cohorts.
DATA_FILTER_EXCEL = "/home/student/lisa_ma/datasets/VerSe/data_filter_joined.xlsx"
ALLOWED_DSNAMES = {"dataset-verse19", "dataset-verse20"}

# Manually reviewed anomaly/mislabel table; patients flagged here are excluded so
# the examples shown are clean, correctly labelled vertebrae.
GT_EXCEL = "/home/student/lisa_ma/evaluation/A_CT_ANOMALY_LABELv9.xlsx"

spec = {
    "directory": "results/VerSe_classifier",
    "dataset_dir": "/home/student/lisa_ma/prepared",
    "job_name": "viz_region_classes",
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


def get_verse_pids(data_filter_excel_path, allowed_dsnames):
    """PIDs whose dsname is in allowed_dsnames (i.e. genuinely from the public
    VerSe dataset, not one of the other clinical source cohorts)."""
    df = pd.read_excel(data_filter_excel_path)
    verse = df[df["dsname"].isin(allowed_dsnames)]
    return set(verse["pid"].astype(str).str.strip())


def get_flagged_mislabel_pids(gt_excel_path):
    """PIDs ('fid') flagged as known mislabels / removals in the reviewed GT excel."""
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


def vert_to_name(vert):
    """Vertebra number -> anatomical label (e.g. 4 -> 'C4', 10 -> 'T3', 20 -> 'L1').
    Matches the numbering used by VerSeDataset.get_region_label_from_vert."""
    if 1 <= vert <= 7:
        return f"C{vert}"
    elif 8 <= vert <= 19:
        return f"T{vert - 7}"
    elif 20 <= vert <= 24:
        return f"L{vert - 19}"
    elif 25 <= vert <= 29:
        return f"S{vert - 24}"
    return "?"


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

    verse_pids = get_verse_pids(DATA_FILTER_EXCEL, ALLOWED_DSNAMES)
    print(f"Restricting selection to {len(verse_pids)} patients from {sorted(ALLOWED_DSNAMES)}")

    flagged_pids = get_flagged_mislabel_pids(GT_EXCEL)
    print(f"Excluding {len(flagged_pids)} patients flagged as known mislabels.")

    # Collect all candidate slices per region (VerSe patients only). Multiple
    # examples from the same patient are allowed. Each entry stores
    # (pid, idx, corrected_vert) so we can label the exact vertebra shown.
    region_candidates = defaultdict(list)  # label -> [(pid, idx, corrected_vert), ...]
    for idx, path in enumerate(dataset.file_paths):
        pid = VerSeDataset.get_pid_from_path(path)
        if pid not in verse_pids or pid in flagged_pids:
            continue
        # Apply the same pid corrections the dataset uses before deriving the region.
        orig_vert = VerSeDataset.get_vert_number(path)
        corrected_vert = dataset.pid_corrections.get(pid, {}).get(orig_vert, orig_vert)
        label = VerSeDataset.get_region_label_from_vert(corrected_vert)
        if label in REGION_NAMES:
            region_candidates[label].append((pid, idx, corrected_vert))

    print("VerSe candidate slices available per region:")
    for label in REGION_ORDER:
        print(f"  {REGION_NAMES[label]}: {len(region_candidates.get(label, []))} slices")

    # Pick EXAMPLES_PER_CLASS examples per region, greedily preferring variety:
    # first unseen vertebra levels, then unseen patients, then anything left.
    chosen = {}  # label -> [(pid, idx, corrected_vert), ...]
    for label in REGION_ORDER:
        candidates = region_candidates.get(label, [])
        if len(candidates) < EXAMPLES_PER_CLASS:
            raise ValueError(
                f"Region '{REGION_NAMES[label]}' only has {len(candidates)} VerSe "
                f"slices (< {EXAMPLES_PER_CLASS} needed)."
            )
        picked, seen_verts, seen_pids = [], set(), set()
        for want_new_vert in (True, False):
            for want_new_pid in (True, False):
                for pid, idx, vert in candidates:
                    if len(picked) >= EXAMPLES_PER_CLASS:
                        break
                    if (pid, idx) in {(p, i) for p, i, _ in picked}:
                        continue
                    if want_new_vert and vert in seen_verts:
                        continue
                    if want_new_pid and pid in seen_pids:
                        continue
                    picked.append((pid, idx, vert))
                    seen_verts.add(vert)
                    seen_pids.add(pid)
        chosen[label] = picked[:EXAMPLES_PER_CLASS]

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    plt.rcParams.update({"font.family": "serif", "font.size": 11})

    n_rows = len(REGION_ORDER)
    fig, axes = plt.subplots(
        n_rows,
        EXAMPLES_PER_CLASS,
        figsize=(2.4 * EXAMPLES_PER_CLASS + 1.2, 2.4 * n_rows),
    )

    for r, label in enumerate(REGION_ORDER):
        for c, (pid, idx, corrected_vert) in enumerate(chosen[label]):
            image_tensor, _, sample_id = dataset[idx]
            display_slice = get_display_slice(image_tensor)

            ax = axes[r, c]
            # aspect="auto" makes each slice fill its cell, so there is no vertical
            # letterboxing between the label and the row below it.
            ax.imshow(display_slice, cmap="gray", aspect="auto")
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)

            # Exact vertebra label (e.g. 'C4', 'T7') below each image.
            ax.set_xlabel(vert_to_name(corrected_vert), fontsize=12, labelpad=3)

            # Row label (class name) to the left of the first column, CIFAR-style.
            if c == 0:
                ax.set_ylabel(
                    REGION_NAMES[label],
                    rotation=0,
                    ha="right",
                    va="center",
                    labelpad=12,
                    fontsize=13,
                    fontweight="bold",
                )

    fig.subplots_adjust(left=0.18, right=0.98, top=0.98, bottom=0.04,
                        wspace=0.01, hspace=0.22)

    grid_path = os.path.join(OUTPUT_DIR, "region_classes_examples.png")
    fig.savefig(grid_path, dpi=300, bbox_inches="tight")
    print(f"\nSaved figure: {grid_path}")
    for label in REGION_ORDER:
        ids = [f"{pid}({vert_to_name(v)})" for pid, _, v in chosen[label]]
        print(f"  {REGION_NAMES[label]}: {ids}")


if __name__ == "__main__":
    main()
