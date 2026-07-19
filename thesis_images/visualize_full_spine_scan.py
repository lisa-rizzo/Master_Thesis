"""
Standalone script to render a full/regional spine CT slice from the raw
(pre-cropping) VerSe scans, to illustrate basic CT windowing for the thesis
theory chapter (bone = bright/high HU, soft tissue = grey/mid HU, air = dark).

These raw scans (BIDS-style rawdata/*_ct.nii.gz) are the original acquisitions
that dataloaders/dataloader_VerSe.py's per-vertebra .npz crops were cut from -
this script reads them directly and does not touch any existing project code.

Run with: python visualize_full_spine_scan.py
"""

import os

import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt

OUTPUT_DIR = "thesis_images"

# HU window applied before display: values <= LEVEL-WIDTH/2 -> black,
# values >= LEVEL+WIDTH/2 -> white. A wide "bone + soft tissue" window
# so the full grey (soft tissue) -> bright (bone) gradient is visible.
WINDOW_LEVEL = 200
WINDOW_WIDTH = 1400

SCANS = [
    {
        "path": "/media/datasets/dataset-verse19/dataset-verse19/rawdata/sub-verse050/sub-verse050_ct.nii.gz",
        "out_name": "full_spine_scan_sagittal",
        "region": "full",  # full sagittal slice, whole torso/spine
    },
    {
        "path": "/media/datasets/dataset-verse19/dataset-verse19/rawdata/sub-verse050/sub-verse050_ct.nii.gz",
        "out_name": "regional_spine_scan_thoracolumbar",
        "region": "crop",  # zoomed-in thoracolumbar region, same subject
    },
]


def apply_window(slice_hu, level, width):
    lo, hi = level - width / 2, level + width / 2
    clipped = np.clip(slice_hu, lo, hi)
    return (clipped - lo) / (hi - lo)


def best_midline_slice(data):
    """Pick the sagittal index (last axis) with the most bone-HU voxels,
    as a proxy for the slice passing through the vertebral column."""
    bone_counts = [(data[:, :, z] > 300).sum() for z in range(data.shape[2])]
    return int(np.argmax(bone_counts))


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    plt.rcParams.update({"font.family": "serif", "font.size": 10})

    for scan in SCANS:
        img = nib.load(scan["path"])
        data = img.get_fdata()

        z = best_midline_slice(data)
        slice_hu = data[:, :, z]
        display_slice = apply_window(slice_hu, WINDOW_LEVEL, WINDOW_WIDTH)

        if scan["region"] == "crop":
            # Zoom into the central region where the spine sits, cutting
            # away empty margins on both spatial axes.
            h, w = display_slice.shape
            display_slice = display_slice[int(h * 0.15):int(h * 0.75), int(w * 0.25):int(w * 0.75)]

        out_path = os.path.join(OUTPUT_DIR, f"{scan['out_name']}.png")
        plt.imsave(out_path, display_slice, cmap="gray", origin="upper")
        print(f"Saved: {out_path}  (slice z={z} of {data.shape[2]}, shape {display_slice.shape})")


if __name__ == "__main__":
    main()
