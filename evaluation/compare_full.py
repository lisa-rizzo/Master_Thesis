"""
Generate a vertebra-specific comparison table of GLARE scores against ground-truth.

Outputs an Excel file with all original columns plus:
  - mislabel_origin ("Confirmed mislabel" = in both GT and GLARE data, "Confirmed (not in training data)" =
    GT-flagged but absent from the training run, "" = not flagged as a mislabel)
  - true_anomaly_label (original vert number before pid_corrections: 20 for vert 19 mislabels, 28 for vert 20 mislabels)
  - class / alternative_class (glare) are written out as anatomical region names
    (Cervical/Thoracic/Lumbar/Sacral) rather than the underlying 0-3 codes.
  - Final column headers: id, counted_vert_number, counted_vert_name, mislabel_origin,
    glare_score, class, alternative_class (glare), true_anomaly_label, true_vert_name,
    exclusion_reason (see RENAME_COLUMNS/COLUMN_ORDER).

Rows where mislabel_origin == ORIGIN_CONFIRMED are colored green.

Ground-truth source: gt_mislabels_complete.xlsx
  - T12_missing: vertebra 19 is mislabeled (gt_label=20), vertebra 24 is mislabeled (gt_label=25)
  - T13_extra:   vertebra 20 is mislabeled (gt_label=28), vertebra 25 is mislabeled (gt_label=24)
"""
from pathlib import Path
import pandas as pd
import re
from datetime import datetime
import logging
from openpyxl import load_workbook
from openpyxl.styles import PatternFill, Font, Alignment
from openpyxl.utils import get_column_letter

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger("compare_mislabels")

# File paths
MISLABELS_CSV = Path("/home/student/lisa_ma/results/glare/22-08-26_11:28_ep-15_dual-gpu/results/mislabels_list_20260822_112835.csv")
GT_EXCEL = Path("/home/student/lisa_ma/evaluation/gt_mislabels_complete.xlsx")
DATA_FILTER_EXCEL = Path("/home/student/lisa_ma/datasets/VerSe/data_filter_joined.xlsx")
OUTPUT_DIR = Path("/home/student/lisa_ma/evaluation")

# Color fills
GREEN_FILL = PatternFill(start_color="90EE90", end_color="90EE90", fill_type="solid")   # Confirmed mislabel, has a GLARE score
GT_ONLY_FILL = PatternFill(start_color="FFD966", end_color="FFD966", fill_type="solid")  # Confirmed, but no GLARE score (not in training data)
GLARE_ONLY_FILL = PatternFill(start_color="ADD8E6", end_color="ADD8E6", fill_type="solid")  # mislabels from the original scan labels

# Header/title styling (shared "brand" blue between the data sheet header and the legend)
HEADER_FILL = PatternFill(start_color="1F4E78", end_color="1F4E78", fill_type="solid")     # dark blue
HEADER_FONT = Font(color="FFFFFF", bold=True, size=13)
LEGEND_TITLE_FILL = PatternFill(start_color="1F4E78", end_color="1F4E78", fill_type="solid")   # dark blue
LEGEND_TITLE_FONT = Font(color="FFFFFF", bold=True, size=18)
LEGEND_SECTION_FILL = PatternFill(start_color="D9E1F2", end_color="D9E1F2", fill_type="solid")  # light blue
LEGEND_SECTION_FONT = Font(bold=True, size=14)
LEGEND_BODY_FONT_SIZE = 13  # legend body text (descriptions), 2 sizes up from Excel's default 11

# Region code -> anatomical region name (used for the "class" and
# "alternative class (glare)" columns)
REGION_NAMES = {0: "Cervical", 1: "Thoracic", 2: "Lumbar", 3: "Sacral"}

# First letter of a vertebra name -> full region name, used to compare a
# naive vs. true region (e.g. counted_vert_name "T12" vs true_vert_name "T13")
REGION_LETTER_TO_NAME = {"C": "Cervical", "T": "Thoracic", "L": "Lumbar", "S": "Sacral"}

# Columns that should be written as real numbers (not text) so Excel can sort/filter numerically
NUMERIC_COLUMNS = ["glare_score", "counted_vert_number", "true_anomaly_label"]

# mislabel_origin category labels (blank = not flagged by any source)
ORIGIN_CONFIRMED = "Confirmed mislabel"
ORIGIN_GT_ONLY = "Confirmed (not in training data)"
ORIGIN_CORRECTION_ONLY = "mislabels from the original scan labels"

# Rename from the internal working names (matched to the source GLARE CSV /
# earlier columns) to the final, human-facing column headers.
RENAME_COLUMNS = {
    "label": "class",
    "vertebra_number": "counted_vert_number",
    "vertebra_name": "counted_vert_name",
    "gt_label": "true_anomaly_label",
    "gt_vertebra_name": "true_vert_name",
    "glare score": "glare_score",
    "alternative class (glare)": "alternative_class (glare)",
}

# Output column order: identify the case, then GLARE's verdict, then the
# ground-truth answer (kept next to GLARE's guess for easy comparison),
# then training/technical metadata last.
COLUMN_ORDER = [
    "id",
    "counted_vert_number",
    "counted_vert_name",
    "mislabel_origin",
    "glare_score",
    "class",
    "alternative_class (glare)",
    "true_anomaly_label",
    "true_vert_name",
    "exclusion_reason",
]

# Columns to exclude from output
EXCLUDE_COLUMNS = ["glarex score", "alternative glare score", "alternative class (glarex)", "alternative glarex score"]


def parse_pid_and_vert(vert_id):
    """
    Extract pid and vertebra number from vertebra_id.
    Format: pid_vert<number>
    Returns: (pid, vert_num) or (None, None) if parsing fails
    """
    match = re.match(r"^(.+?)_vert(\d+)$", str(vert_id))
    if match:
        return match.group(1), int(match.group(2))
    return None, None


def get_vertebra_name(vert_num):
    """
    Map a vertebra number to its anatomical designation.
    1-7 -> C1-C7, 8-19 -> T1-T12, 20-24 -> L1-L5, 25-29 -> S1-S5.
    Vertebra 28 is the overflow slot used for an additional thoracic
    vertebra (T13) in the corrected gt_label numbering scheme.
    """
    try:
        v = int(vert_num)
    except (TypeError, ValueError):
        return ""
    if 1 <= v <= 7:
        return f"C{v}"
    elif 8 <= v <= 19:
        return f"T{v - 7}"
    elif 20 <= v <= 24:
        return f"L{v - 19}"
    elif v == 28:
        return "T13"
    elif 25 <= v <= 29:
        return f"S{v - 24}"
    else:
        return str(v)


def get_region_name(code):
    """
    Map a region code (0=Cervical, 1=Thoracic, 2=Lumbar, 3=Sacral) to its
    anatomical name, for the "class" and "alternative class (glare)"
    columns. Blank/unparseable values are returned as an empty string.
    """
    if code is None or str(code).strip() == "" or str(code).strip().lower() == "nan":
        return ""
    try:
        c = int(float(code))
    except (TypeError, ValueError):
        return str(code)
    return REGION_NAMES.get(c, str(code))


def get_gt_label(vert_num):
    """
    Get the correct ground-truth label for a mislabeled vertebra.
    - Vertebra 19 mislabel → gt_label = 20
    - Vertebra 20 mislabel → gt_label = 28
    - All others → gt_label = vert_num
    """
    if vert_num == 19:
        return "20"
    elif vert_num == 20:
        return "28"
    else:
        return str(vert_num)


def load_data_filter(path):
    """Load the training-data filter sheet (has_T1 / is_consecutive / split per PID)."""
    return pd.read_excel(path, dtype=str, keep_default_na=False)


def get_exclusion_reason(pid, df_filter):
    """
    Explain why a ground-truth-flagged PID has no glare_score, by checking
    the same has_T1/is_consecutive/split filter the training dataloader
    applies before a scan is ever loaded (see get_data_VerSe.py).
    """
    matches = df_filter[df_filter["pid"].astype(str).str.startswith(pid)]
    if len(matches) == 0:
        return "PID not found in the training data-filter sheet (data_filter_joined.xlsx)"

    row = matches.iloc[0]
    has_t1 = str(row.get("has_T1", "")).strip()
    is_consecutive = str(row.get("is_consecutive", "")).strip()
    split = str(row.get("split", "")).strip()

    if has_t1 != "1":
        return "Excluded before training: has_T1 = 0 (no clear C7/T1 landmark, e.g. partial field of view)"
    if is_consecutive != "1":
        return "Excluded before training: is_consecutive = 0 in the data filter"
    if split != "train":
        return f"Excluded before training: PID is in the '{split}' split (GLARE only analyzed the 'train' split)"
    return ("Passes all training-data filters (has_T1=1, is_consecutive=1, train split) - likely a missing or "
            "incomplete file on disk rather than a filter exclusion")


def parse_vertebra_list(value):
    """
    Parse a raw LabelOverride cell (e.g. "[9, 10, 11, ..., 23]") into a
    list of vertebra-number ints, robust to brackets/whitespace.
    """
    return [int(n) for n in re.findall(r"\d+", str(value))]


def compare_label_sequences(training_verts, override_verts):
    """
    Compare a training vertebra-number sequence with the override
    (expert-corrected) vertebra-number sequence, position by position.
    Both arguments are sequences of ints, in vertebra order.

    Returns: set of positions (0-based index into the sequences) that
    are mislabeled.

    Logic:
    - Find first difference between sequences
    - Mark that position and all subsequent positions where the two
      sequences still differ (a resync stops the flagging)
    - Any position beyond the shorter sequence's length is also marked
    """
    mislabeled_positions = set()

    if not training_verts or not override_verts:
        return mislabeled_positions

    # Find first difference
    first_diff_idx = None
    for i in range(min(len(training_verts), len(override_verts))):
        if training_verts[i] != override_verts[i]:
            first_diff_idx = i
            break

    if first_diff_idx is not None:
        mislabeled_positions.add(first_diff_idx)

        # Mark all subsequent shifted positions
        for i in range(first_diff_idx + 1, min(len(training_verts), len(override_verts))):
            if training_verts[i] != override_verts[i]:
                mislabeled_positions.add(i)

    # Handle length mismatch
    if len(training_verts) != len(override_verts):
        max_len = max(len(training_verts), len(override_verts))
        min_len = min(len(training_verts), len(override_verts))
        for i in range(min_len, max_len):
            mislabeled_positions.add(i)

    return mislabeled_positions


def get_region_letter(name):
    """First letter of a vertebra name (e.g. 'T12' -> 'T'), used to compare
    a naive vs. true region without needing the full region-name string."""
    s = str(name).strip()
    if not s or s.lower() == "nan":
        return ""
    return s[0].upper()


def compute_statistics(df_out, thresholds=(5, 7, 10, 12, 13, 14)):
    """
    Compute the analysis metrics shown on the Statistics sheet.

    Boolean aggregation is done via plain Python lists/ints, not pandas
    columns - mixing None and bool in an object-dtype pandas column can
    silently break .mean() (this produced a wrong "100%" figure once
    already, corrected here).
    """
    def is_flagged(v):
        return isinstance(v, str) and v.strip() != ""

    scored = df_out[df_out["glare_score"].notna()].copy()
    scored["pid"] = scored["id"].apply(lambda x: parse_pid_and_vert(x)[0])
    scored["vnum"] = scored["id"].apply(lambda x: parse_pid_and_vert(x)[1])

    confirmed = df_out[df_out["mislabel_origin"] == ORIGIN_CONFIRMED].copy()
    confirmed["pid"] = confirmed["id"].apply(lambda x: parse_pid_and_vert(x)[0])
    confirmed["vnum"] = confirmed["id"].apply(lambda x: parse_pid_and_vert(x)[1])
    confirmed["naive_letter"] = confirmed["counted_vert_name"].apply(get_region_letter)
    confirmed["true_letter"] = confirmed["true_vert_name"].apply(get_region_letter)
    confirmed["true_region_name"] = confirmed["true_letter"].map(REGION_LETTER_TO_NAME)

    crossing = confirmed[confirmed["naive_letter"] != confirmed["true_letter"]].copy()
    within = confirmed[confirmed["naive_letter"] == confirmed["true_letter"]].copy()

    n_crossing_correct = int((crossing["alternative_class (glare)"] == crossing["true_region_name"]).sum())
    n_within_correct = int((within["alternative_class (glare)"] == within["true_region_name"]).sum())

    clean_scored = df_out[(df_out["mislabel_origin"] == "") & df_out["glare_score"].notna()]
    threshold_rows = []
    for t in thresholds:
        recall = (int((crossing["glare_score"] <= t).sum()) / len(crossing)) if len(crossing) else 0.0
        fpr = (int((clean_scored["glare_score"] <= t).sum()) / len(clean_scored)) if len(clean_scored) else 0.0
        threshold_rows.append((t, recall, fpr))

    # Anchor-point precision: for each edge-case row, compare its glare_score
    # against every other scored vertebra in the same PID (same scan).
    is_global_min_flags = []
    is_local_min_flags = []
    non_min_winner_flagged = []  # only for rows that are NOT the in-scan minimum
    for _, r in crossing.iterrows():
        pid, vnum, score = r["pid"], r["vnum"], r["glare_score"]
        if pid is None or vnum is None or pd.isna(score):
            continue
        scan = scored[scored["pid"] == pid]
        if scan.empty:
            continue
        min_score = float(scan["glare_score"].min())
        at_min = bool(float(score) <= min_score)
        is_global_min_flags.append(at_min)

        neighbors = scan[(scan["vnum"] >= vnum - 2) & (scan["vnum"] <= vnum + 2) & (scan["vnum"] != vnum)]
        beats_neighbors = True if neighbors.empty else bool(float(score) <= float(neighbors["glare_score"].min()))
        is_local_min_flags.append(beats_neighbors)

        if not at_min:
            winners = scan[scan["glare_score"] == min_score]
            non_min_winner_flagged.append(bool(winners["mislabel_origin"].apply(is_flagged).any()))

    n_not_min = len(non_min_winner_flagged)
    n_not_min_flagged = sum(1 for x in non_min_winner_flagged if x)

    return {
        "n_crossing": len(crossing),
        "n_within": len(within),
        "mean_crossing": float(crossing["glare_score"].mean()) if len(crossing) else None,
        "mean_within": float(within["glare_score"].mean()) if len(within) else None,
        "n_crossing_correct": n_crossing_correct,
        "n_within_correct": n_within_correct,
        "threshold_rows": threshold_rows,
        "n_global_min": sum(1 for x in is_global_min_flags if x),
        "n_total_anchor": len(is_global_min_flags),
        "n_local_min": sum(1 for x in is_local_min_flags if x),
        "n_not_min": n_not_min,
        "n_not_min_flagged": n_not_min_flagged,
        "n_not_min_clean": n_not_min - n_not_min_flagged,
        "crossing_ids": set(crossing["id"]),
    }


def add_legend_sheet(wb, stats=None, generated_from=None):
    """
    Add a "Legend" sheet, inserted as the first tab, explaining every
    column and the row-color coding for anyone opening the file who isn't
    familiar with how it's built (e.g. Hendrik).
    """
    ws = wb.create_sheet(title="Legend", index=0)
    col_b_width = 100
    ws.column_dimensions["A"].width = 32
    ws.column_dimensions["B"].width = col_b_width
    ws.sheet_view.showGridLines = False

    wrap = Alignment(wrap_text=True, vertical="top")
    body_font = Font(size=LEGEND_BODY_FONT_SIZE)
    # Body text is bigger than the Excel default (11pt) this was originally tuned for,
    # so fewer characters fit per wrapped line, and each line needs more vertical space.
    body_chars_per_line = max(20, int(col_b_width * 11 / LEGEND_BODY_FONT_SIZE))
    body_line_height = 15 * LEGEND_BODY_FONT_SIZE / 11

    def row_height_for(text, min_height=24):
        if not text:
            return min_height
        # Account for explicit line breaks (e.g. paragraph spacing) in addition
        # to auto-wrapping, so multi-paragraph descriptions don't get clipped.
        total_lines = sum(max(1, -(-len(p) // body_chars_per_line)) for p in str(text).split("\n"))
        return max(min_height, int(total_lines * body_line_height + 8))

    def write_row(r, a, b="", section=False, title=False, fill=None):
        cell_a = ws.cell(row=r, column=1, value=a)
        cell_b = ws.cell(row=r, column=2, value=b)
        cell_a.alignment = wrap
        cell_b.alignment = wrap
        if title:
            cell_a.font = LEGEND_TITLE_FONT
            cell_b.font = LEGEND_TITLE_FONT
            cell_a.fill = LEGEND_TITLE_FILL
            cell_b.fill = LEGEND_TITLE_FILL
            ws.row_dimensions[r].height = 36
        elif section:
            cell_a.font = LEGEND_SECTION_FONT
            cell_b.font = LEGEND_SECTION_FONT
            cell_a.fill = LEGEND_SECTION_FILL
            cell_b.fill = LEGEND_SECTION_FILL
            ws.row_dimensions[r].height = 28
        else:
            cell_a.font = body_font
            cell_b.font = body_font
            if fill:
                cell_a.fill = fill
                cell_b.fill = fill
            ws.row_dimensions[r].height = row_height_for(b)

    row = 1
    write_row(row, "Full_Comparison sheet - column guide", title=True)
    ws.merge_cells(start_row=row, start_column=1, end_row=row, end_column=2)
    row += 2

    if generated_from:
        write_row(row, "Generated", generated_from.get("timestamp", ""))
        row += 1
        write_row(row, "Ground-truth sheet", generated_from.get("gt_sheet", ""))
        row += 2

    if stats:
        write_row(row, "Summary", "Count", section=True)
        row += 1
        write_row(row, "Total rows", stats["total_rows"])
        row += 1
        write_row(row, "Confirmed mislabels from the anomaly excel", stats["confirmed"])
        row += 1
        write_row(row, "Mislabels from the original scan labels", stats["correction_only"])
        row += 1
        write_row(row, "Confirmed but not in training data", stats["confirmed_not_in_training"])
        row += 1
        write_row(row, "Not flagged", stats["not_flagged"])
        row += 2

    write_row(row, "Column", "Description", section=True)
    row += 1
    # Kept in the same left-to-right order as the actual sheet (see COLUMN_ORDER)
    columns = [
        ("id", 'PID + vertebra number as originally labeled, e.g. "pid_vert19"'),
        ("counted_vert_number", "The raw vertebra number as originally counted top-to-bottom, before any "
                                 "correction - what was assumed, not necessarily what's actually true."),
        ("counted_vert_name", "Anatomical name for counted_vert_number, e.g. T12, L1 "),
        ("mislabel_origin", "How this row's mislabel status was determined - see categories below."),
        ("exclusion_reason", "Only filled in for rows with no glare_score (i.e. mislabel_origin = "
                              f'"{ORIGIN_GT_ONLY}" or "{ORIGIN_CORRECTION_ONLY}" without training data). Explains, '
                              "using the same has_T1/is_consecutive/split filter the training pipeline applies "
                              "before loading a scan, exactly why this vertebra was never analyzed by GLARE - "
                              "confirming it's a structural gap, not a computation error."),
        ("glare_score", "GLARE score (0-15): number of training epochs where the assigned label had the lowest "
                         "gradient norm. Lower Glare Score = more likely a mislabel."),
        ("class", "The anatomical region actually used to train the classifier: Cervical, Thoracic, Lumbar, or "
                  "Sacral. Included mainly for reference."),
        ("alternative_class (glare)", "The anatomical region (Cervical/Thoracic/Lumbar/Sacral) GLARE considers the "
                                       "next most likely correct class, if not the assigned one - i.e. GLARE's own "
                                       "guess at the fix. Compare this against true_vert_name (a few columns over) "
                                       "to see whether GLARE's guess matches the actual confirmed answer."),
        ("true_anomaly_label", "The corrected/true vertebra number, according to an independent review - filled in "
                                "only when that review disagrees with counted_vert_number. This is the answer to "
                                "check GLARE's guess against."),
        ("true_vert_name", "Anatomical name for true_anomaly_label."),
    ]
    for name, desc in columns:
        write_row(row, name, desc)
        row += 1

    row += 1
    write_row(row, "mislabel_origin value", "Meaning", section=True)
    row += 1
    categories = [
        ("(blank)", "Not flagged as a mislabel by any source - treated as clean/correctly labeled."),
        (ORIGIN_CONFIRMED,
            "The strongest category: from gt_mislabels_complete.xlsx, which lists all vertebrae that receive "
            "a wrong region label in training due to pid_corrections (top-down counting applied for T12-missing "
            "and T13-extra transitional anatomy cases). The vertebra was also part of the training run GLARE "
            "analyzed, so it has a real glare score. These are the rows to use for judging GLARE itself: does "
            "a low glare_score / does the alternative_class actually line up with true_anomaly_label?"),
        (ORIGIN_GT_ONLY,
            "Also confirmed as a real mislabel by gt_mislabels_complete.xlsx - but this vertebra was never part "
            "of the training run GLARE analyzed (e.g. it's in the validation/test data split, or it was excluded "
            "by the has_T1/is_consecutive data filters before training even started). Because GLARE never saw "
            "this sample, there is no glare score to check - the blank cells here are expected, not missing data."),
    ]
    for name, desc in categories:
        write_row(row, name, desc)
        row += 1

    row += 1
    write_row(row, "Row color", "Meaning", section=True)
    row += 1
    write_row(row, "Green", ORIGIN_CONFIRMED, fill=GREEN_FILL)
    row += 1
    write_row(row, "Amber", ORIGIN_GT_ONLY, fill=GT_ONLY_FILL)


def add_statistics_sheet(wb, stats):
    """
    Add a "Statistics" sheet with the aggregate analysis metrics (region-
    crossing vs. within-region breakdown, ceiling effect, anchor-point
    precision, threshold recall/FPR table, anomaly-type x source
    breakdown). Row-level data lives in "Edge Cases" instead.
    """
    ws = wb.create_sheet(title="Statistics")
    widths = [40, 22, 22, 45]
    for col_idx, w in enumerate(widths, start=1):
        ws.column_dimensions[get_column_letter(col_idx)].width = w
    ws.sheet_view.showGridLines = False

    wrap = Alignment(wrap_text=True, vertical="top")
    body_font = Font(size=LEGEND_BODY_FONT_SIZE)
    body_chars_per_line = max(20, int(sum(widths) * 11 / LEGEND_BODY_FONT_SIZE))
    body_line_height = 15 * LEGEND_BODY_FONT_SIZE / 11

    def row_height_for(text, min_height=24):
        if not text:
            return min_height
        total_lines = sum(max(1, -(-len(p) // body_chars_per_line)) for p in str(text).split("\n"))
        return max(min_height, int(total_lines * body_line_height + 8))

    def write_row(r, *values, section=False, title=False, note=False):
        cells = [ws.cell(row=r, column=c, value=v) for c, v in enumerate(values, start=1) if v is not None]
        for cell in cells:
            cell.alignment = wrap
        if title:
            for cell in cells:
                cell.font = LEGEND_TITLE_FONT
                cell.fill = LEGEND_TITLE_FILL
            ws.row_dimensions[r].height = 36
        elif section:
            for cell in cells:
                cell.font = LEGEND_SECTION_FONT
                cell.fill = LEGEND_SECTION_FILL
            ws.row_dimensions[r].height = 28
        else:
            for cell in cells:
                cell.font = body_font
            longest = max((str(v) for v in values if v is not None), key=len, default="")
            ws.row_dimensions[r].height = row_height_for(longest) if note else 24

    row = 1
    write_row(row, "Statistics - GLARE Evaluation Summary", title=True)
    ws.merge_cells(start_row=row, start_column=1, end_row=row, end_column=4)
    row += 1
    write_row(row, "See the Legend tab for column definitions and category meanings.")
    row += 2

    write_row(row, "Region-Crossing vs. Within-Region Mislabels (Confirmed mislabel only, n="
                   f"{stats['n_crossing'] + stats['n_within']})", section=True)
    row += 1
    write_row(row, "Category", "Count", "Mean glare_score", "alt_class matches true region", section=True)
    row += 1
    write_row(row, "Region-crossing (edge case)", stats["n_crossing"], round(stats["mean_crossing"], 1),
              f"{stats['n_crossing_correct']}/{stats['n_crossing']} ({stats['n_crossing_correct']/stats['n_crossing']:.1%})")
    row += 1
    write_row(row, "Within-region", stats["n_within"], round(stats["mean_within"], 1),
              f"{stats['n_within_correct']}/{stats['n_within']} ({stats['n_within_correct']/stats['n_within']:.1%}) -> is expected")
    row += 1
    write_row(row, "> cascading-relabeling only needs the boundary vertebra detected; everything downstream can be "
                   "corrected by shifting", note=True)
    row += 2

    write_row(row, f"Anchor-Point Precision (edge-case rows only, n={stats['n_total_anchor']})", section=True)
    row += 1
    write_row(row, "Metric", "Count", "Percent", section=True)
    row += 1
    write_row(row, "Is the single lowest-scoring vertebra in its scan ",
              f"{stats['n_global_min']}/{stats['n_total_anchor']}", f"{stats['n_global_min']/stats['n_total_anchor']:.1%}")
    row += 1
    write_row(row, "Beats its immediate +/-2-vertebra neighbors specifically",
              f"{stats['n_local_min']}/{stats['n_total_anchor']}", f"{stats['n_local_min']/stats['n_total_anchor']:.1%}")
    row += 1
    write_row(row, "Of the non-lowest rows: loses to another real, flagged anomaly in the same scan",
              f"{stats['n_not_min_flagged']}/{stats['n_not_min']}",
              f"{stats['n_not_min_flagged']/stats['n_not_min']:.1%}" if stats['n_not_min'] else "n/a")
    row += 1
    write_row(row, "Of the non-lowest rows: loses to a genuinely clean/unflagged vertebra (a real false alarm)",
              f"{stats['n_not_min_clean']}/{stats['n_not_min']}",
              f"{stats['n_not_min_clean']/stats['n_not_min']:.1%}" if stats['n_not_min'] else "n/a")
    row += 1
    write_row(row, "This matters for the cascading-relabeling: picking 'the lowest-scoring vertebra in a scan' as "
                   "the anchor point almost always lands on a real anomaly - but not always the specific one you "
                   "were checking, since some scans have multiple simultaneous anomalies competing for the lowest "
                   "score, and a small fraction of the time a genuinely clean vertebra scores even lower than the "
                   "true anomaly", note=True)
    row += 2

    write_row(row, "Threshold-Based Recall / False-Positive Rate (edge-cases vs. clean rows)", section=True)
    row += 1
    write_row(row, "glare_score <=", "Recall on edge-cases", "False-positive rate on clean rows", section=True)
    row += 1
    for threshold, recall, fpr in stats["threshold_rows"]:
        write_row(row, threshold, f"{recall:.1%}", f"{fpr:.1%}")
        row += 1


def main():
    # Check files exist
    if not MISLABELS_CSV.exists():
        log.error("Mislabels CSV not found: %s", MISLABELS_CSV)
        return
    if not GT_EXCEL.exists():
        log.error("Ground-truth Excel not found: %s", GT_EXCEL)
        return

    log.info("Reading mislabels CSV (full dataset): %s", MISLABELS_CSV)
    df_mis = pd.read_csv(MISLABELS_CSV, dtype=str, keep_default_na=False)
    log.info("Total rows: %d", len(df_mis))

    log.info("Reading ground-truth Excel: %s", GT_EXCEL)
    df_gt = pd.read_excel(GT_EXCEL, dtype=str, keep_default_na=False)
    log.info("Ground-truth rows: %d", len(df_gt))

    log.info("Reading training data-filter sheet: %s", DATA_FILTER_EXCEL)
    df_data_filter = load_data_filter(DATA_FILTER_EXCEL)
    log.info("Data-filter rows: %d", len(df_data_filter))

    # Build vertebra-specific ground-truth mislabel set from gt_mislabels_complete.xlsx.
    # Format: {(pid, vert_num): gt_label}  where gt_label is the original (pre-correction) vert number.
    gt_mislabeled_verts = {}

    for _, row in df_gt.iterrows():
        pid = str(row["pid"]).strip()
        vert_id = int(row["vert_id"])
        anomaly_type = str(row.get("anomaly_type", "")).strip()

        # Derive the true vert number (before pid_corrections shifted it):
        # T12_missing Rule B shifted verts >=20 down by 1  → true vert = vert_id + 1
        # T13_extra  Rule A shifted vert28→20             → true vert = 28
        #            Rule A shifted vert24→25             → true vert = 24
        if anomaly_type == "T12_missing":
            gt_label = str(vert_id + 1)
        elif anomaly_type == "T13_extra":
            if vert_id == 20:
                gt_label = "28"
            elif vert_id == 25:
                gt_label = "24"
            else:
                gt_label = str(vert_id)
        else:
            gt_label = str(vert_id)

        gt_mislabeled_verts[(pid, vert_id)] = gt_label
        log.debug("%s for %s: marking vert %d (gt=%s)", anomaly_type, pid, vert_id, gt_label)

    log.info("Ground-truth mislabeled vertebrae: %d", len(gt_mislabeled_verts))

    # No separate removed_labels set — gt_mislabels_complete is the single source of truth.
    removed_labels_verts = {}

    # Process each row in the full dataset
    output_rows = []

    for _, row in df_mis.iterrows():
        # Get all original columns except excluded ones
        row_dict = {}
        for col in row.index:
            if col not in EXCLUDE_COLUMNS:
                row_dict[col] = row[col]
        
        vert_id = str(row["id"])
        pid, vert_num = parse_pid_and_vert(vert_id)
        
        # Initialize new columns
        mislabel_origin = ""
        gt_label = ""

        if pid and vert_num is not None:
            in_gt = (pid, vert_num) in gt_mislabeled_verts
            in_removed = (pid, vert_num) in removed_labels_verts

            # Determine mislabel origin
            if in_gt:
                mislabel_origin = ORIGIN_CONFIRMED
                gt_label = gt_mislabeled_verts[(pid, vert_num)]
            elif in_removed:
                # In removed labels but not GT
                mislabel_origin = ORIGIN_CORRECTION_ONLY
                gt_label = removed_labels_verts[(pid, vert_num)]

        # Add new columns to the row
        row_dict["vertebra_number"] = vert_num
        row_dict["vertebra_name"] = get_vertebra_name(vert_num)
        row_dict["mislabel_origin"] = mislabel_origin
        row_dict["exclusion_reason"] = ""  # this row has data, so it was never excluded
        row_dict["gt_label"] = gt_label
        row_dict["gt_vertebra_name"] = get_vertebra_name(gt_label) if gt_label else ""

        output_rows.append(row_dict)
    
    # Add GT-only and removed-only vertebrae that don't exist in the dataset
    existing_verts = set()
    for _, row in df_mis.iterrows():
        pid, vert_num = parse_pid_and_vert(row["id"])
        if pid and vert_num is not None:
            existing_verts.add((pid, vert_num))
    
    # GT-only vertebrae
    for (pid, vert_num), gt_lbl in gt_mislabeled_verts.items():
        if (pid, vert_num) not in existing_verts:
            output_rows.append({
                "id": f"{pid}_vert{vert_num}",
                "label": "",
                "glare score": "",
                "alternative class (glare)": "",
                "vertebra_number": vert_num,
                "vertebra_name": get_vertebra_name(vert_num),
                "mislabel_origin": ORIGIN_GT_ONLY,
                "exclusion_reason": get_exclusion_reason(pid, df_data_filter),
                "gt_label": gt_lbl,
                "gt_vertebra_name": get_vertebra_name(gt_lbl),
            })
    
    # Removed-only vertebrae (not in GT or existing data)
    for (pid, vert_num), gt_lbl in removed_labels_verts.items():
        if (pid, vert_num) not in existing_verts and (pid, vert_num) not in gt_mislabeled_verts:
            output_rows.append({
                "id": f"{pid}_vert{vert_num}",
                "label": "",
                "glare score": "",
                "alternative class (glare)": "",
                "vertebra_number": vert_num,
                "vertebra_name": get_vertebra_name(vert_num),
                "mislabel_origin": ORIGIN_CORRECTION_ONLY,
                "exclusion_reason": get_exclusion_reason(pid, df_data_filter),
                "gt_label": gt_lbl,
                "gt_vertebra_name": get_vertebra_name(gt_lbl),
            })
    
    df_out = pd.DataFrame(output_rows)
    df_out = df_out.rename(columns=RENAME_COLUMNS)
    df_out = df_out[COLUMN_ORDER]

    # Write numeric-looking columns as real numbers (not text), so Excel's
    # sort/filter treats them numerically instead of alphabetically
    for col in NUMERIC_COLUMNS:
        df_out[col] = pd.to_numeric(df_out[col], errors="coerce")

    # Region codes -> anatomical names, for readability
    df_out["class"] = df_out["class"].apply(get_region_name)
    df_out["alternative_class (glare)"] = df_out["alternative_class (glare)"].apply(get_region_name)

    # Summary stats
    n_match = len(df_out[df_out["mislabel_origin"] == ORIGIN_CONFIRMED])
    n_gt_only = len(df_out[df_out["mislabel_origin"] == ORIGIN_GT_ONLY])
    n_glare_only = len(df_out[df_out["mislabel_origin"] == ORIGIN_CORRECTION_ONLY])
    n_no_mislabel = len(df_out[df_out["mislabel_origin"] == ""])

    log.info("Dataset statistics:")
    log.info("  - Total rows: %d", len(df_out))
    log.info("  - Not marked as mislabel: %d", n_no_mislabel)
    log.info("  - In both GT and GLARE (match): %d", n_match)
    log.info("  - Only in GT (gt): %d", n_gt_only)
    log.info("  - Only in GLARE (glare): %d", n_glare_only)

    summary_stats = {
        "total_rows": len(df_out),
        "confirmed": n_match,
        "confirmed_not_in_training": n_gt_only,
        "correction_only": n_glare_only,
        "not_flagged": n_no_mislabel,
    }
    analysis_stats = compute_statistics(df_out)
    combined_stats = {**summary_stats, **analysis_stats}

    # Sort by glare_score (ascending, most suspicious first) as the primary
    # key. Tiebreak by mislabel_origin category (keeps flagged rows grouped
    # within the large tie-block at the score ceiling), then by id for full
    # run-to-run determinism. Rows with no glare_score sort last.
    origin_priority = {ORIGIN_CONFIRMED: 0, ORIGIN_CORRECTION_ONLY: 1, ORIGIN_GT_ONLY: 2, "": 3}
    df_out["_origin_priority"] = df_out["mislabel_origin"].map(origin_priority).fillna(3)
    df_out = df_out.sort_values(by=["glare_score", "_origin_priority", "id"], na_position="last")
    df_out = df_out.drop(columns="_origin_priority").reset_index(drop=True)

    # Save to Excel with color coding
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = OUTPUT_DIR / f"full_comparison_{ts}.xlsx"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Write to Excel
    df_out.to_excel(output_path, index=False, sheet_name="Full_Comparison")
    log.info("Wrote comparison table to: %s", output_path)

    # Apply formatting: header style, column widths, filter/sort, freeze panes, row colors
    wb = load_workbook(output_path)
    ws = wb.active

    # Find column indices
    header_row = [cell.value for cell in ws[1]]
    try:
        origin_col_idx = header_row.index("mislabel_origin") + 1
    except ValueError as e:
        log.warning("Could not find required columns for highlighting: %s", e)
        wb.save(output_path)
        return

    # Header row: bold white-on-blue, bigger font/row, frozen so it stays visible while scrolling
    for col_idx in range(1, ws.max_column + 1):
        cell = ws.cell(row=1, column=col_idx)
        cell.fill = HEADER_FILL
        cell.font = HEADER_FONT
    ws.row_dimensions[1].height = 26
    ws.freeze_panes = "B2"  # keep both the header row and the id column visible while scrolling

    # Enable Excel's built-in filter/sort dropdowns on every column
    ws.auto_filter.ref = ws.dimensions

    # Column widths, sized to the widest value actually present in each column.
    # Extra padding (+7 instead of just enough to fit the text) leaves room for
    # the autofilter dropdown arrow next to the header without it overlapping the title.
    for col_idx, col_name in enumerate(df_out.columns, start=1):
        values = df_out[col_name].astype(str)
        max_len = max([len(str(col_name))] + [len(v) for v in values if v and v != "nan"])
        ws.column_dimensions[get_column_letter(col_idx)].width = min(max_len + 7, 50)

    # Color rows (starting from row 2, after header)
    # green = confirmed mislabel, amber = known GT mislabel with no GLARE
    # score, blue = mislabels from the original scan labels
    for row_idx in range(2, ws.max_row + 1):
        origin_value = ws.cell(row=row_idx, column=origin_col_idx).value

        if origin_value == ORIGIN_CONFIRMED:
            fill = GREEN_FILL
        elif origin_value == ORIGIN_GT_ONLY:
            fill = GT_ONLY_FILL
        elif origin_value == ORIGIN_CORRECTION_ONLY:
            fill = GLARE_ONLY_FILL
        else:
            continue  # No coloring

        for col_idx in range(1, ws.max_column + 1):
            ws.cell(row=row_idx, column=col_idx).fill = fill

    # "Edge Cases" sheet: only the region-crossing Confirmed mislabel rows -
    # the ones GLARE can actually be meaningfully evaluated against. df_out
    # is already sorted by glare_score, so filtering preserves that order.
    edge_ws = wb.create_sheet(title="Edge Cases")
    edge_df = df_out[df_out["id"].isin(combined_stats["crossing_ids"])]
    for r_idx, row_vals in enumerate([list(df_out.columns)] + edge_df.values.tolist(), start=1):
        for c_idx, val in enumerate(row_vals, start=1):
            edge_ws.cell(row=r_idx, column=c_idx, value=None if pd.isna(val) else val)
    for col_idx in range(1, edge_ws.max_column + 1):
        cell = edge_ws.cell(row=1, column=col_idx)
        cell.fill = HEADER_FILL
        cell.font = HEADER_FONT
    edge_ws.row_dimensions[1].height = 26
    edge_ws.freeze_panes = "B2"
    edge_ws.auto_filter.ref = edge_ws.dimensions
    for col_idx, col_name in enumerate(df_out.columns, start=1):
        values = edge_df[col_name].astype(str)
        max_len = max([len(str(col_name))] + [len(v) for v in values if v and v != "nan"])
        edge_ws.column_dimensions[get_column_letter(col_idx)].width = min(max_len + 7, 50)
    log.info("Added Edge Cases sheet: %d rows", len(edge_df))

    add_statistics_sheet(wb, combined_stats)
    log.info("Added Statistics sheet")

    generated_from = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "gt_sheet": GT_EXCEL.name,
    }
    add_legend_sheet(wb, stats=combined_stats, generated_from=generated_from)
    wb.active = wb.sheetnames.index("Legend")

    wb.save(output_path)
    log.info("Applied formatting: header style, column widths, autofilter, frozen header row")
    log.info("Row colors: green=%s, amber=%s, blue=%s",
              ORIGIN_CONFIRMED, ORIGIN_GT_ONLY, ORIGIN_CORRECTION_ONLY)
    log.info("Added Legend sheet (first tab)")
    log.info("Done! Output file: %s", output_path)

    # Preview
    print("\nPreview (first 10 rows with mislabels):")
    mislabeled_rows = df_out[df_out["mislabel_origin"] != ""]
    if len(mislabeled_rows) > 0:
        print(mislabeled_rows.head(10).to_string(index=False))
    else:
        print("No mislabels found in dataset")


if __name__ == "__main__":
    main()