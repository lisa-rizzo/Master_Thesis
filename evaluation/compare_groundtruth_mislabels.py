"""
Generate a comparison table of low-confidence GLARE mislabels against ground-truth.

Outputs an Excel file with:
  - vertebra_id (from mislabels CSV or GT)
  - training_label (label column from mislabels)
  - glare_score
  - alternative_class
  - mislabel_origin (0 = only in GT, 1 = in both, 2 = only in GLARE mislabels)
  - gt_label (label from GT table if found)
  
Rows where mislabel_origin == 1 are colored green.
"""
from pathlib import Path
import pandas as pd
import re
from datetime import datetime
import logging
from openpyxl import load_workbook
from openpyxl.styles import PatternFill

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger("compare_mislabels")

# File paths
MISLABELS_CSV = Path("/home/student/lisa_ma/results/glare/13-10-25_15:18_ep-15_single-gpu/mislabels_only_20251013_151808.csv")
GT_EXCEL = Path("/home/student/lisa_ma/evaluation/A_CT_ANOMALY_LABELv9.xlsx")
OUTPUT_DIR = Path("/home/student/lisa_ma/evaluation")

# Thresholds and column names
GLARE_THRESHOLD = 7.0
GREEN_FILL = PatternFill(start_color="90EE90", end_color="90EE90", fill_type="solid")


def parse_pid(vert_id):
    """
    Extract pid from vertebra_id, ignoring any suffix.
    Format: pid_vert<number> or pid_ses-YYYYMMDD_ce-xxx_vert<number>
    Returns: pid or None if parsing fails
    """
    match = re.match(r"^(.+?)_vert\d+$", str(vert_id))
    if match:
        return match.group(1)
    return None


def main():
    # Check files exist
    if not MISLABELS_CSV.exists():
        log.error("Mislabels CSV not found: %s", MISLABELS_CSV)
        return
    if not GT_EXCEL.exists():
        log.error("Ground-truth Excel not found: %s", GT_EXCEL)
        return

    log.info("Reading mislabels CSV: %s", MISLABELS_CSV)
    df_mis = pd.read_csv(MISLABELS_CSV, dtype=str, keep_default_na=False)
    log.info("Total mislabels rows: %d", len(df_mis))

    log.info("Reading ground-truth Excel: %s", GT_EXCEL)
    df_gt = pd.read_excel(GT_EXCEL, dtype=str, keep_default_na=False)
    log.info("Ground-truth rows: %d", len(df_gt))

    # Filter ground-truth for actual mislabels (T11 == "1" OR T13 == "1" OR LabelOverride is not empty)
    df_gt_mislabels = df_gt[
        (df_gt.get("T11") == "1") |
        (df_gt.get("T13") == "1") |
        ((df_gt.get("LabelOverride", "").astype(str).str.strip() != "") &
         (df_gt.get("LabelOverride", "").notna()))
    ].copy()
    log.info("Ground-truth entries with mislabels (T11=1 or T13=1 or LabelOverride): %d", len(df_gt_mislabels))

    # Extract PIDs from filtered ground-truth
    mislabel_pids = set()
    for _, row in df_gt_mislabels.iterrows():
        # fid column likely doesn't have _vert_xx suffix, so just use it directly
        pid = str(row["fid"]).strip()
        if pid:
            mislabel_pids.add(pid)

    log.info("Unique mislabel PIDs in ground-truth: %d", len(mislabel_pids))

    # Identify columns in mislabels CSV
    id_col = "id"
    label_col = "label"
    glare_col = "glare score"
    alt_class_col = "alternative class (glare)"

    # Check required columns exist
    required_cols = [id_col, label_col, glare_col, alt_class_col]
    missing = [c for c in required_cols if c not in df_mis.columns]
    if missing:
        log.error("Missing columns in mislabels CSV: %s", missing)
        log.error("Available columns: %s", list(df_mis.columns))
        return

    # Convert glare score to numeric and filter
    df_mis[glare_col] = pd.to_numeric(df_mis[glare_col], errors='coerce')
    df_filtered = df_mis[df_mis[glare_col] <= GLARE_THRESHOLD].copy()
    log.info("Mislabels with glare score <= %s: %d", GLARE_THRESHOLD, len(df_filtered))

    # Build output table - first from GLARE mislabels
    output_rows = []
    glare_pids = set()
    
    for _, row in df_filtered.iterrows():
        vert_id = str(row[id_col])
        train_label = str(row[label_col])
        glare_score = row[glare_col]
        alt_class = str(row[alt_class_col])

        # Parse pid from vertebra_id
        pid = parse_pid(vert_id)
        if pid:
            glare_pids.add(pid)

        # Check if pid is in mislabel PIDs
        # 1 = in both, 2 = only in GLARE mislabels
        mislabel_origin = 1 if pid in mislabel_pids else 2

        output_rows.append({
            "vertebra_id": vert_id,
            "training_label": train_label,
            "glare_score": glare_score,
            "alternative_class": alt_class,
            "mislabel_origin": mislabel_origin,
            "gt_label": ""
        })

    # Add ground-truth mislabels that were NOT found in GLARE predictions
    gt_only_pids = mislabel_pids - glare_pids
    log.info("Ground-truth mislabels NOT found by GLARE: %d PIDs", len(gt_only_pids))
    
    for _, row in df_gt_mislabels.iterrows():
        pid = str(row["fid"]).strip()
        if pid in gt_only_pids:
            # This is a ground-truth mislabel not found by GLARE
            output_rows.append({
                "vertebra_id": pid,  # Use the fid as vertebra_id
                "training_label": "",
                "glare_score": "",
                "alternative_class": "",
                "mislabel_origin": 0,  # Only in ground truth
                "gt_label": ""
            })

    df_out = pd.DataFrame(output_rows)
    
    # Summary stats
    n_both = len(df_out[df_out["mislabel_origin"] == 1])
    n_gt_only = len(df_out[df_out["mislabel_origin"] == 0])
    n_glare_only = len(df_out[df_out["mislabel_origin"] == 2])
    
    log.info("Mislabel origin breakdown:")
    log.info("  - In both GT and GLARE (origin=1): %d", n_both)
    log.info("  - Only in GT (origin=0): %d", n_gt_only)
    log.info("  - Only in GLARE (origin=2): %d", n_glare_only)
    log.info("  - Total rows: %d", len(df_out))

    # Save to Excel with color coding
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = OUTPUT_DIR / f"mislabels_comparison_{ts}.xlsx"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Write to Excel
    df_out.to_excel(output_path, index=False, sheet_name="Comparison")
    log.info("Wrote comparison table to: %s", output_path)

    # Apply green highlighting to rows where mislabel_origin == 1
    wb = load_workbook(output_path)
    ws = wb.active

    # Find mislabel_origin column index (1-based for openpyxl)
    header_row = [cell.value for cell in ws[1]]
    try:
        origin_col_idx = header_row.index("mislabel_origin") + 1
    except ValueError:
        log.warning("Could not find 'mislabel_origin' column for highlighting.")
        wb.save(output_path)
        return

    # Color rows where mislabel_origin == 1 (starting from row 2, after header)
    for row_idx in range(2, ws.max_row + 1):
        cell_value = ws.cell(row=row_idx, column=origin_col_idx).value
        if cell_value == 1:
            # Color all vertebra_ids of that pid
            pid = parse_pid(ws.cell(row=row_idx, column=1).value)  # Assuming vertebra_id is in the first column
            if pid:
                for r in range(2, ws.max_row + 1):
                    r_pid = parse_pid(ws.cell(row=r, column=1).value)
                    if r_pid == pid:
                        for col_idx in range(1, ws.max_column + 1):
                            ws.cell(row=r, column=col_idx).fill = GREEN_FILL

    wb.save(output_path)
    log.info("Applied green highlighting to rows with mislabel_origin=1")
    log.info("Done! Output file: %s", output_path)

    # Preview
    print("\nPreview (first 10 rows):")
    print(df_out.head(10).to_string(index=False))


if __name__ == "__main__":
    main()