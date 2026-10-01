"""
Central path resolution.

Everything inside the repo is resolved relative to the repo root. The dataset comes from
spec["dataset_dir"] or, if set, from the environment variable MISLABELDET_DATASET_DIR
(which takes precedence — see portable_spec).
"""

import os
from pathlib import Path

# helper/paths.py -> repo root
REPO_ROOT = Path(__file__).resolve().parent.parent

# --- Repo-internal files ---
EXCEL_FILTER = REPO_ROOT / "data" / "data_filter_joined.xlsx"
PID_CORRECTIONS_JSON = REPO_ROOT / "data" / "pid_corrections.json"
GT_EXCEL = REPO_ROOT / "evaluation" / "A_CT_ANOMALY_LABELv9.xlsx"
# Ground truth: 109 cross-region mislabels, one row per sample
# (sample_id, anomaly_type, training_class, true_class, ...).
GT_REGION_EXCEL = REPO_ROOT / "evaluation" / "gt_mislabels_complete.xlsx"

# --- Dataset (default if neither the environment variable nor spec["dataset_dir"] is set) ---
DATASET_DIR = Path(os.environ.get("MISLABELDET_DATASET_DIR", "/home/student/lisa_ma/prepared"))


def portable_spec(spec: dict) -> dict:
    """
    Make a spec.json from another machine runnable here (in place, returns spec).

    - dataset_dir: MISLABELDET_DATASET_DIR wins if set, else the value from the spec, else the default.
    - excel_path: the absolute path stored in the spec points to the checkout where
      training ran; if it does not exist here, the repo copy is used instead.
    """
    env = os.environ.get("MISLABELDET_DATASET_DIR")
    spec["dataset_dir"] = env or spec.get("dataset_dir") or str(DATASET_DIR)
    excel = spec.get("excel_path")
    if not excel or not Path(excel).is_file():
        spec["excel_path"] = str(EXCEL_FILTER)
    return spec
