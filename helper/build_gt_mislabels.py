"""
build_gt_mislabels.py — Derive mislabeled training sample IDs from source data.

Reconstructs the ground-truth mislabel list by applying the same labeling pipeline
as dataloader_VerSe_new.py and reasoning from the anomaly Excel about where the
resulting training labels are anatomically wrong.

Scope: T13_extra and T12_missing anomalies only (C/T boundary crossings excluded).

Domain rules used (confirmed):
  T13_extra — A scan with an extra thoracic vertebra.
    The T13 vertebra is always Thoracic (true_class=Thoracic), regardless of how
    it was numbered by the annotator. If it ends up at training vert 20 (Lumbar label),
    it is mislabeled. If the L5 is consequently pushed to training vert 25 (Sacral label),
    that is also mislabeled (true_class=Lumbar).
  T12_missing — A scan where T12 is absent.
    L1 is always Lumbar. If it ends up at training vert 19 (Thoracic label), mislabeled.
    If S1 is pushed to training vert 24 (Lumbar label), also mislabeled (true_class=Sacral).
  LabelOverride — The override vert (from LO step-1, before Rule A/B compose) is the
    ground-truth vertebra identity. region_label(override_vert) is the true class,
    with the exception of vert 28 (VerSe code for T13), whose true class is Thoracic.

Mislabel sources:
  A. T13_extra non-LO: PIDs with T13==1 & Sanity==1 in anomaly Excel, or in pid_fix_28
     (T13 annotated as vert28). LO PIDs excluded — handled by Source C.
     → training vert 20 (Lumbar, true Thoracic) and vert 25 if present (Sacral, true Lumbar)
  B. T12_missing: all pid_fix_1820 PIDs (includes those that also have LO, since the
     T12_missing identity is certain regardless of LO remapping).
     → training vert 19 (Thoracic, true Lumbar) and vert 24 if present (Lumbar, true Sacral)
  C. LabelOverride: for each LO PID, compare true_class=region_label(override_vert)
     with training_class=region_label(training_vert). A mislabel exists when they differ
     at a canonical boundary position:
       T13_extra: override_vert=28 → training_vert=20, OR override_vert=24 → training_vert=25
       T12_missing: override_vert=20 → training_vert=19, OR override_vert=25 → training_vert=24
     (Only canonical pairs are tracked; intermediate Rule-A/B-shifted verts are out of scope.
      T12_missing LO entries already covered by Source B are skipped to avoid duplicates.)

Output columns: sample_id, pid, dsname, vert_id, training_class, true_class,
                anomaly_type, source, disk_vert_id

Usage:
  python build_gt_mislabels.py [--spec SPEC] [--filter FILTER] [--anomaly ANOMALY] [--out OUT]
"""

import argparse
import ast
import json
from pathlib import Path

import pandas as pd

DEFAULT_SPEC    = "results/VerSe_classifier/label_override_8ep_23092026_2135/spec.json"
DEFAULT_FILTER  = "data/data_filter_joined.xlsx"
DEFAULT_ANOMALY = "evaluation/A_CT_ANOMALY_LABELv9.xlsx"
DEFAULT_OUT     = "evaluation/gt_mislabels_derived.csv"

INT_TO_CLASS = {0: "Cervical", 1: "Thoracic", 2: "Lumbar", 3: "Sacral"}

def region_label(vert: int) -> int:
    """Same as VerSeDataset.get_region_label_from_vert."""
    if 1  <= vert <= 7:  return 0
    if 8  <= vert <= 19: return 1
    if 20 <= vert <= 24: return 2
    if 25 <= vert <= 29: return 3
    return -1

def true_region_label(vert: int) -> int:
    """True anatomical region label. vert 28 = T13 (Thoracic), not Sacral."""
    if vert == 28:
        return 1  # T13 is Thoracic
    return region_label(vert)

def apply_rules(verts: set[int]) -> dict[int, int]:
    """Rule A / Rule B — mirrors _compute_pid_corrections."""
    if 28 in verts and 20 in verts:
        return {v: (20 if v == 28 else (v + 1 if v >= 20 else v)) for v in verts}
    if 20 in verts and 19 not in verts and 18 in verts:
        return {v: (v - 1 if v >= 20 else v) for v in verts}
    return {v: v for v in verts}

def compose_lo_with_rules(lo_step1: dict[int, int]) -> dict[int, int]:
    """LO step-2 — mirrors _compose_override_with_rules."""
    ov = set(lo_step1.values())
    if 28 in ov and 20 in ov:
        rule = {v: (20 if v == 28 else (v + 1 if v >= 20 else v)) for v in ov}
    elif 20 in ov and 19 not in ov and 18 in ov:
        rule = {v: (v - 1 if v >= 20 else v) for v in ov}
    else:
        rule = {v: v for v in ov}
    return {dv: rule.get(ov_v, ov_v) for dv, ov_v in lo_step1.items()}

def parse_verts(val) -> set[int]:
    if isinstance(val, str):
        return set(int(v) for v in ast.literal_eval(val) if pd.notna(v))
    if hasattr(val, "__iter__"):
        return set(int(v) for v in val if pd.notna(v))
    return {int(val)}


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--spec",    default=DEFAULT_SPEC)
    parser.add_argument("--filter",  default=DEFAULT_FILTER)
    parser.add_argument("--anomaly", default=DEFAULT_ANOMALY)
    parser.add_argument("--out",     default=DEFAULT_OUT)
    args = parser.parse_args()

    # ── Spec: pid_fix lists ───────────────────────────────────────────────────
    with open(args.spec) as f:
        spec = json.load(f)
    pid_fix_1820: set[str] = set(spec.get("pid_fix_1820", []))
    pid_fix_28:   set[str] = set(spec.get("pid_fix_28",   []))
    print(f"pid_fix_1820: {len(pid_fix_1820)}  pid_fix_28: {len(pid_fix_28)}")

    # ── Filter Excel: training PIDs + disk vert sets ──────────────────────────
    fdf = pd.read_excel(args.filter)
    fdf = fdf.loc[(fdf["has_T1"] == 1) & (fdf["is_consecutive"] == 1)]
    pid_info: dict[str, dict] = {
        str(r["pid"]).strip(): {
            "dsname":     str(r["dsname"]).strip(),
            "split":      r["split"],
            "disk_verts": parse_verts(r["vert_label"]),
        }
        for _, r in fdf.iterrows()
    }
    train_pids = {p for p, v in pid_info.items() if v["split"] == "train"}
    print(f"Training PIDs: {len(train_pids)}")

    # ── Anomaly Excel ─────────────────────────────────────────────────────────
    adf = pd.read_excel(args.anomaly)

    # T13_extra PIDs: T13==1 AND Sanity==1 AND not removed
    t13_pids: set[str] = {
        str(r["fid"]).strip()
        for _, r in adf.iterrows()
        if r.get("T13", 0) == 1 and r.get("Sanity", 0) == 1 and r.get("Remove", 0) != 1
    }
    print(f"T13_extra PIDs (T13==1, Sanity==1): {len(t13_pids)}")

    # LabelOverride step-1 maps: disk_vert → override_vert (positional)
    lo_step1_all: dict[str, dict[int, int]] = {}
    for _, r in adf.iterrows():
        if pd.isna(r.get("LabelOverride")):
            continue
        pid = str(r["fid"]).strip()
        if pid not in pid_info:
            continue
        seq = (ast.literal_eval(r["LabelOverride"])
               if isinstance(r["LabelOverride"], str)
               else list(r["LabelOverride"]))
        seq = [int(v) for v in seq]
        disk_sorted = sorted(pid_info[pid]["disk_verts"])
        if len(disk_sorted) != len(seq):
            print(f"  [LO] length mismatch for {pid} — skip")
            continue
        lo_step1_all[pid] = {d: c for d, c in zip(disk_sorted, seq)}
    lo_pids = set(lo_step1_all.keys())
    print(f"LabelOverride PIDs: {len(lo_pids)}")

    # ── Build pid_corrections (Rule A/B for pid_fix PIDs, LO overrides) ───────
    pid_corrections: dict[str, dict[int, int]] = {}
    for pid in pid_fix_1820 | pid_fix_28:
        if pid in pid_info:
            pid_corrections[pid] = apply_rules(pid_info[pid]["disk_verts"])
    for pid, step1 in lo_step1_all.items():
        pid_corrections[pid] = compose_lo_with_rules(step1)

    # ── Collect mislabels ─────────────────────────────────────────────────────
    rows = []

    def identity(pid):
        return {v: v for v in pid_info[pid]["disk_verts"]}

    # ── Source A: T13_extra non-LO ────────────────────────────────────────────
    # T13 ends up at training vert 20 (Lumbar label, true Thoracic=T13).
    # L5 ends up at training vert 25 if present (Sacral label, true Lumbar=L5).
    # Covers both sub-cases:
    #   A1. T13 annotated directly as vert20 (anomaly Excel T13==1, Sanity==1):
    #       no Rule A fires, training vert 20 == disk vert 20.
    #   A2. T13 annotated as vert28 (pid_fix_28 PIDs):
    #       Rule A fires: disk28 → training20, disk24 → training25.
    for pid in ((t13_pids | pid_fix_28) - lo_pids) & train_pids:
        corr   = pid_corrections.get(pid, identity(pid))
        tverts = set(corr.values())
        inv    = {tv: dv for dv, tv in corr.items()}
        dsname = pid_info[pid]["dsname"]
        if 20 in tverts:
            rows.append(dict(sample_id=f"{pid}_vert20", pid=pid, dsname=dsname,
                             vert_id=20, training_class="Lumbar", true_class="Thoracic",
                             anomaly_type="T13_extra", source="original",
                             disk_vert_id=inv.get(20) if inv.get(20) != 20 else None))
        if 25 in tverts:
            rows.append(dict(sample_id=f"{pid}_vert25", pid=pid, dsname=dsname,
                             vert_id=25, training_class="Sacral", true_class="Lumbar",
                             anomaly_type="T13_extra", source="original",
                             disk_vert_id=inv.get(25) if inv.get(25) != 25 else None))

    # ── Source B: T12_missing ─────────────────────────────────────────────────
    # All pid_fix_1820 PIDs, including those that also have LO. The T12_missing
    # identity is certain regardless of LO remapping: any vert at training position 19
    # is L1 (Lumbar) mislabeled as Thoracic, and any vert at training position 24 is
    # S1 (Sacral) mislabeled as Lumbar.
    for pid in pid_fix_1820 & train_pids:
        corr   = pid_corrections.get(pid, identity(pid))
        tverts = set(corr.values())
        dsname = pid_info[pid]["dsname"]
        if 19 in tverts:
            rows.append(dict(sample_id=f"{pid}_vert19", pid=pid, dsname=dsname,
                             vert_id=19, training_class="Thoracic", true_class="Lumbar",
                             anomaly_type="T12_missing", source="original",
                             disk_vert_id=None))
        if 24 in tverts:
            rows.append(dict(sample_id=f"{pid}_vert24", pid=pid, dsname=dsname,
                             vert_id=24, training_class="Lumbar", true_class="Sacral",
                             anomaly_type="T12_missing", source="original",
                             disk_vert_id=None))

    # ── Source C: LabelOverride ───────────────────────────────────────────────
    # For each LO PID: compare true_class=true_region_label(override_vert) with
    # training_class=region_label(training_vert) at canonical boundary positions.
    #
    # T13_extra: only the T13 vert itself (override28 → training20) and the
    #   L5 cascade (override24 → training25) are canonical T13_extra mislabels.
    #   Intermediate verts shifted by Rule A (e.g. T11 at training24) are out of scope.
    #
    # T12_missing: only L1 (override20 → training19) and S1 (override25 → training24).
    #   PIDs already covered by Source B are skipped to avoid duplicate entries.
    for pid in lo_pids & train_pids:
        step1    = lo_step1_all[pid]
        composed = pid_corrections[pid]
        dsname   = pid_info[pid]["dsname"]

        for disk_v, training_v in composed.items():
            override_v = step1[disk_v]

            # T13_extra: T13 (override28) pushed to training vert 20
            if override_v == 28 and training_v == 20:
                rows.append(dict(sample_id=f"{pid}_vert20", pid=pid, dsname=dsname,
                                 vert_id=20, training_class="Lumbar", true_class="Thoracic",
                                 anomaly_type="T13_extra", source="LabelOverride",
                                 disk_vert_id=disk_v))

            # T13_extra: L5 (override24) pushed to training vert 25
            elif override_v == 24 and training_v == 25:
                rows.append(dict(sample_id=f"{pid}_vert25", pid=pid, dsname=dsname,
                                 vert_id=25, training_class="Sacral", true_class="Lumbar",
                                 anomaly_type="T13_extra", source="LabelOverride",
                                 disk_vert_id=disk_v))

            # T12_missing: L1 (override20) pushed to training vert 19
            # Skip if already covered by Source B (pid in pid_fix_1820)
            elif override_v == 20 and training_v == 19 and pid not in pid_fix_1820:
                rows.append(dict(sample_id=f"{pid}_vert19", pid=pid, dsname=dsname,
                                 vert_id=19, training_class="Thoracic", true_class="Lumbar",
                                 anomaly_type="T12_missing", source="LabelOverride",
                                 disk_vert_id=disk_v))

            # T12_missing: S1 (override25) pushed to training vert 24
            elif override_v == 25 and training_v == 24 and pid not in pid_fix_1820:
                rows.append(dict(sample_id=f"{pid}_vert24", pid=pid, dsname=dsname,
                                 vert_id=24, training_class="Lumbar", true_class="Sacral",
                                 anomaly_type="T12_missing", source="LabelOverride",
                                 disk_vert_id=disk_v))

    # ── Known data errors: manually exclude ──────────────────────────────────
    # ced0209_ses-20130703_sequ-2_ce-no_vert20:
    #   The LabelOverride column in the anomaly Excel maps disk19 → vert28 (T13),
    #   but the same row has T13=0 and HE Comment="Rippen, bleibt T12" — the
    #   annotator saw a rib but explicitly decided it is T12, not T13. The disk
    #   sequence [8..23] has no vert28. The LO entry is a data entry error in the
    #   anomaly Excel. Removing to avoid a false GT mislabel.
    KNOWN_ERRORS = {"ced0209_ses-20130703_sequ-2_ce-no_vert20"}
    rows = [r for r in rows if r["sample_id"] not in KNOWN_ERRORS]

    # ── Output ────────────────────────────────────────────────────────────────
    COLS = ["sample_id", "pid", "dsname", "vert_id", "training_class", "true_class",
            "anomaly_type", "source", "disk_vert_id"]
    out_df = (pd.DataFrame(rows, columns=COLS)
                .drop_duplicates(subset=["sample_id"])
                .sort_values(["anomaly_type", "pid", "vert_id"])
                .reset_index(drop=True))

    print(f"\nTotal mislabeled training samples: {len(out_df)}")
    print(out_df["anomaly_type"].value_counts().to_string())
    print(out_df.groupby(["anomaly_type", "source"]).size().to_string())

    out_df.to_csv(args.out, index=False)
    print(f"\nSaved to: {args.out}")

    # ── Compare against existing GT file (diagnostic only) ────────────────────
    ref_path = Path("evaluation/gt_mislabels_complete_LO.xlsx")
    if ref_path.exists():
        ref = pd.read_excel(ref_path)
        derived_ids = set(out_df["sample_id"])
        ref_ids     = set(ref["sample_id"].astype(str))
        only_derived = derived_ids - ref_ids
        only_ref     = ref_ids - derived_ids
        print(f"\nDiagnostic comparison against {ref_path.name}:")
        print(f"  Derived: {len(derived_ids)}  |  Existing GT: {len(ref_ids)}")
        if only_derived:
            print(f"  In derived only — may indicate GT file is missing entries:")
            for s in sorted(only_derived):
                print(f"    {s}")
        if only_ref:
            print(f"  In existing GT only — may indicate GT file has extra/wrong entries:")
            for s in sorted(only_ref):
                print(f"    {s}")
        if not only_derived and not only_ref:
            print("  Derived and existing GT are identical.")


if __name__ == "__main__":
    main()
