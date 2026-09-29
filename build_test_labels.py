"""
build_test_labels.py — Build the ground-truth test label CSV.

Three steps based on the old dataloader (dataloader_VerSe.py):

  Step 1 — Anomaly corrections from A_CT_ANOMALY_LABELv9.xlsx
    T13 at V:  disk V  -> 28        (insert T13 label)
               disk D > V -> D-1    (shift cascade down by 1)
    T11 at V:  disk D >= V -> D+1   (shift cascade up by 1; "cut" the gap at V)
    (Currently no T11 entries exist in the excel.)

  Step 2 — LabelOverride: positional sequence replacement
    The LabelOverride column holds the complete correct vert sequence for a PID.
    Applied positionally to the on-disk files sorted by disk vert.
    Overwrites any Step-1 correction for that PID.

  Step 3 — Region mapping per PID based on final vert sequence
    Standard:  1-7 Cervical | 8-19 Thoracic | 20-24 Lumbar | 25+ Sacral
    Special:   28 -> Thoracic  (T13 slot, sits between 19 and 20)

Usage:
  python build_test_labels.py
  python build_test_labels.py --out evaluation/test_labels.csv
"""

import argparse
import ast
import sys
from collections import defaultdict
from pathlib import Path

import pandas as pd

sys.path.insert(0, "/home/student/lisa_ma")
import datasets.VerSe.get_data_VerSe as get_data_VerSe
from dataloaders.dataloader_VerSe import VerSeDataset

DATASET_DIR   = "/home/student/lisa_ma/prepared"
FILTER_EXCEL  = "/home/student/lisa_ma/datasets/VerSe/data_filter_joined.xlsx"
ANOMALY_EXCEL = "/home/student/lisa_ma/evaluation/A_CT_ANOMALY_LABELv9.xlsx"
DEFAULT_OUT   = "/home/student/lisa_ma/evaluation/test_labels.csv"

INT_TO_CLASS = {0: "Cervical", 1: "Thoracic", 2: "Lumbar", 3: "Sacral"}


def get_region_label(vert: int) -> int:
    if 1 <= vert <= 7:
        return 0  # Cervical
    if 8 <= vert <= 19 or vert == 28:
        return 1  # Thoracic (28 = T13)
    if 20 <= vert <= 24:
        return 2  # Lumbar
    if vert >= 25:
        return 3  # Sacral
    return -1


def build_step1_map(anomaly_excel: str) -> dict[str, dict[int, int]]:
    """
    Step 1: returns pid -> {disk_vert: corrected_vert} for T13/T11 anomaly entries.
    T13 at V: disk V -> 28, disk D > V -> D-1.
    T11 at V: disk D >= V -> D+1.
    """
    df = pd.read_excel(anomaly_excel)
    if "Remove" in df.columns:
        df = df[df["Remove"] != 1]

    result: dict[str, dict[int, int]] = {}
    for _, row in df.iterrows():
        fid = row.get("fid")
        if pd.isna(fid):
            continue
        pid = str(fid)
        t13 = row.get("T13", 0)
        t11 = row.get("T11", 0)
        is_t13 = not pd.isna(t13) and int(t13) == 1
        is_t11 = not pd.isna(t11) and int(t11) == 1
        raw_v = row.get("vert")
        if pd.isna(raw_v) or not (is_t13 or is_t11):
            continue
        V = int(raw_v)
        result[pid] = {"_type": "T13" if is_t13 else "T11", "_V": V}

    return result


def build_step2_map(anomaly_excel: str,
                    pid_to_sorted_verts: dict[str, list[int]]) -> dict[str, dict[int, int]]:
    """
    Step 2: returns pid -> {disk_vert: correct_vert} using the LabelOverride column.
    The LO sequence is applied positionally to the sorted on-disk vert list.
    """
    df = pd.read_excel(anomaly_excel)
    if "Remove" in df.columns:
        df = df[df["Remove"] != 1]

    lo_df = df[df["LabelOverride"].notna()]
    result: dict[str, dict[int, int]] = {}

    for _, row in lo_df.iterrows():
        fid = row.get("fid")
        if pd.isna(fid):
            continue
        pid = str(fid)
        if pid not in pid_to_sorted_verts:
            continue

        raw = row["LabelOverride"]
        lo_seq = ast.literal_eval(raw) if isinstance(raw, str) else list(raw)
        disk_verts = pid_to_sorted_verts[pid]

        if len(disk_verts) != len(lo_seq):
            print(f"  Warning: {pid} has {len(disk_verts)} disk verts "
                  f"but LabelOverride has {len(lo_seq)} entries — skipping LO.")
            continue

        result[pid] = {dv: int(lv) for dv, lv in zip(disk_verts, lo_seq)}

    return result


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--dataset_dir",   default=DATASET_DIR)
    parser.add_argument("--filter_excel",  default=FILTER_EXCEL)
    parser.add_argument("--anomaly_excel", default=ANOMALY_EXCEL)
    parser.add_argument("--out",           default=DEFAULT_OUT)
    args = parser.parse_args()

    # ── Collect test-split files ───────────────────────────────────────────────
    test_files, missing = get_data_VerSe.get_filtered_files_across_dsnames(
        root_dir=args.dataset_dir,
        excel_path=args.filter_excel,
        split="test",
        glob_pattern="*.npz",
        pid_col="pid",
        dsname_col="dsname",
        verts_col="vert_label",
        check_complete=True,
        apply_excel_filter=True,
        extra_sacral_verts=frozenset(),
    )
    print(f"Test files collected: {len(test_files)}")
    if missing:
        for ds, mp in missing.items():
            for pid, verts in mp.items():
                print(f"  Missing: {ds}/{pid}: {verts}")

    # ── Group on-disk verts per PID (sorted) ──────────────────────────────────
    pid_to_sorted_verts: dict[str, list[int]] = defaultdict(list)
    for p in test_files:
        pid = VerSeDataset.get_pid_from_path(p)
        v   = VerSeDataset.get_vert_number(p)
        if v >= 0:
            pid_to_sorted_verts[pid].append(v)
    for pid in pid_to_sorted_verts:
        pid_to_sorted_verts[pid].sort()

    # ── Step 1: T13/T11 anomaly corrections ───────────────────────────────────
    step1_info = build_step1_map(args.anomaly_excel)
    print(f"Step-1 anomaly PIDs (T13/T11): {len(step1_info)}")

    # ── Step 2: LabelOverride (positional, overwrites Step 1) ─────────────────
    step2_map = build_step2_map(args.anomaly_excel, pid_to_sorted_verts)
    print(f"Step-2 LabelOverride PIDs: {len(step2_map)}")

    # ── Build label rows ───────────────────────────────────────────────────────
    rows    = []
    skipped = 0
    counts  = {"none": 0, "T13": 0, "T11": 0, "LO": 0}

    for p in test_files:
        pid       = VerSeDataset.get_pid_from_path(p)
        disk_vert = VerSeDataset.get_vert_number(p)
        if pid is None or disk_vert < 0:
            skipped += 1
            continue

        # Step 2 takes priority over Step 1
        if pid in step2_map:
            corrected = step2_map[pid].get(disk_vert, disk_vert)
            tag = "LO"
        elif pid in step1_info:
            info = step1_info[pid]
            V    = info["_V"]
            if info["_type"] == "T13":
                if disk_vert == V:
                    corrected = 28
                elif disk_vert > V:
                    corrected = disk_vert - 1
                else:
                    corrected = disk_vert
            else:  # T11
                corrected = disk_vert + 1 if disk_vert >= V else disk_vert
            tag = info["_type"]
        else:
            corrected = disk_vert
            tag = "none"

        label_int = get_region_label(corrected)
        if label_int < 0:
            skipped += 1
            continue

        counts[tag] += 1
        dsname = p.parent.parent.name
        rows.append({
            "sample_id":    f"{pid}_vert{corrected}",
            "pid":          pid,
            "dsname":       dsname,
            "disk_vert_id": disk_vert,
            "vert_id":      corrected,
            "label_int":    label_int,
            "label_str":    INT_TO_CLASS[label_int],
            "correction":   tag,
        })

    if skipped:
        print(f"Skipped {skipped} files (bad filename or unmappable vert).")
    print(f"Corrections — T13: {counts['T13']}, T11: {counts['T11']}, "
          f"LO: {counts['LO']}, none: {counts['none']}")

    # ── Save ───────────────────────────────────────────────────────────────────
    COLS = ["sample_id", "pid", "dsname", "disk_vert_id", "vert_id",
            "label_int", "label_str", "correction"]
    df = (
        pd.DataFrame(rows, columns=COLS)
        .sort_values(["dsname", "pid", "vert_id"])
        .reset_index(drop=True)
    )

    print(f"\nTest samples: {len(df)}")
    print("Class distribution:")
    print(df["label_str"].value_counts().sort_index().to_string())

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"\nSaved to: {out_path}")


if __name__ == "__main__":
    main()
