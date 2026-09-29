"""
Evaluation of Lower Bound and Upper Bound classifiers on the anomaly-corrected test set.

Corrections applied (no Rule A/B):
  T13 at V: disk V -> 28, disk D>V -> D-1
  T11 at V: disk D>=V -> D+1
  LO PIDs:  disk sequence replaced by LabelOverride sequence (overwrites T13/T11)

Region mapping: 1-7=Cervical, 8-19+28=Thoracic, 20-24=Lumbar, 25+=Sacral
"""
import sys
sys.path.insert(0, "/home/student/lisa_ma")

import json
import ast
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, matthews_corrcoef, accuracy_score
from monai.transforms import Compose, NormalizeIntensity

import datasets.VerSe.mi as get_data_VerSe
from dataloaders.dataloader_VerSe import VerSeDataset, custom_collate
from models.model_densenet import DenseNetModel

DATASET_DIR   = "/home/student/lisa_ma/prepared"
FILTER_EXCEL  = "/home/student/lisa_ma/datasets/VerSe/data_filter_joined.xlsx"
ANOMALY_EXCEL = "/home/student/lisa_ma/evaluation/A_CT_ANOMALY_LABELv9.xlsx"
SPEC_DATA     = {"mean": [-111.3885], "std": [406.3665]}
DEVICE        = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")

# ── Region mapping (vert 28 = T13 = Thoracic) ────────────────────────────────
def get_region_label_corrected(vert: int) -> int:
    if 1 <= vert <= 7:
        return 0
    elif (8 <= vert <= 19) or vert == 28:
        return 1
    elif 20 <= vert <= 24:
        return 2
    elif vert >= 25:
        return 3
    return -1

# ── Normalisation transform ───────────────────────────────────────────────────
class Normalize:
    def __init__(self):
        self.t = Compose([NormalizeIntensity(
            subtrahend=SPEC_DATA["mean"], divisor=SPEC_DATA["std"], channel_wise=True
        )])
    def __call__(self, x):
        return self.t(x)

_transform = Normalize()

# ── Load test files ───────────────────────────────────────────────────────────
test_files, _ = get_data_VerSe.get_filtered_files_across_dsnames(
    root_dir=DATASET_DIR, excel_path=FILTER_EXCEL, split="test",
    glob_pattern="*.npz", pid_col="pid", dsname_col="dsname",
    verts_col="vert_label", check_complete=True, apply_excel_filter=True,
    extra_sacral_verts=frozenset(),
)
print(f"Test files: {len(test_files)}")

# ── Build anomaly lookup {pid: {T13, T11, vert}} ──────────────────────────────
filter_df  = pd.read_excel(FILTER_EXCEL)
anomaly_df = pd.read_excel(ANOMALY_EXCEL)

# Drop rows flagged for removal
if "Remove" in anomaly_df.columns:
    anomaly_df = anomaly_df[anomaly_df["Remove"] != 1]

anomaly_lookup: dict[str, dict] = {}
for _, row in anomaly_df.iterrows():
    fid     = row.get("fid")
    dataset = row.get("dataset")
    if pd.isna(fid) or pd.isna(dataset):
        continue
    t13 = row.get("T13", 0); is_T13 = False if pd.isna(t13) else int(t13) == 1
    t11 = row.get("T11", 0); is_T11 = False if pd.isna(t11) else int(t11) == 1
    raw_v  = row.get("vert")
    vert   = int(raw_v) if (raw_v is not None and not pd.isna(raw_v)) else None
    if not (is_T13 or is_T11) or vert is None:
        continue
    pid = str(fid)
    if pid not in anomaly_lookup:
        anomaly_lookup[pid] = {"T13": is_T13, "T11": is_T11, "vert": vert}

print(f"Anomaly PIDs (T13 or T11, non-removed): {len(anomaly_lookup)}")

# ── Load LabelOverride map ────────────────────────────────────────────────────
lo_map  = get_data_VerSe.load_label_override_step1(FILTER_EXCEL, ANOMALY_EXCEL)
lo_pids = set(lo_map.keys())
print(f"LabelOverride PIDs: {len(lo_pids)}")

# ── Sanity: count corrected samples in test set ───────────────────────────────
_n_t13 = _n_t11 = _n_lo = 0
for f in test_files:
    pid = VerSeDataset.get_pid_from_path(f)
    if pid in anomaly_lookup and anomaly_lookup[pid]["T13"]: _n_t13 += 1
    if pid in anomaly_lookup and anomaly_lookup[pid]["T11"]: _n_t11 += 1
    if pid in lo_pids: _n_lo += 1
print(f"Test samples affected: T13={_n_t13}, T11={_n_t11}, LO={_n_lo}")

# ── Custom dataset ────────────────────────────────────────────────────────────
class AnomalyTestDataset(VerSeDataset):
    def __init__(self, file_paths, transform, anomaly_lookup, lo_map):
        super().__init__(file_paths, transform)
        self._anomaly_lookup = anomaly_lookup
        self._lo_map         = lo_map

    def __getitem__(self, idx):
        path = self.file_paths[idx]
        data = np.load(path)
        image_tensor = torch.tensor(data["img"], dtype=torch.float32).unsqueeze(0)
        if self.transform:
            image_tensor = self.transform(image_tensor)

        disk_vert = self.get_vert_number(path)
        pid       = self.get_pid_from_path(path)
        corrected = disk_vert

        # T13 / T11 correction (for non-LO PIDs the LO step below will overwrite anyway)
        if pid in self._anomaly_lookup:
            info = self._anomaly_lookup[pid]
            V = info["vert"]
            if info["T13"]:
                if disk_vert == V:
                    corrected = 28
                elif disk_vert > V:
                    corrected = disk_vert - 1
                # disk_vert < V: unchanged
            elif info["T11"]:
                if disk_vert >= V:
                    corrected = disk_vert + 1
                # disk_vert < V: unchanged

        # LabelOverride overwrites T13/T11
        if pid in self._lo_map:
            corrected = self._lo_map[pid].get(disk_vert, disk_vert)

        label     = get_region_label_corrected(corrected)
        sample_id = f"{pid}_vert{corrected}"
        return image_tensor, label, sample_id

# ── Evaluation helper ─────────────────────────────────────────────────────────
def evaluate(name, spec_path, dataset, epoch=7):
    spec_path = Path(spec_path)
    with open(spec_path) as f:
        spec = json.load(f)
    weights_dir = spec_path.parent / "weights"
    checkpoint  = weights_dir / f"epoch_{epoch}"

    model = DenseNetModel(num_classes=4, spec=spec, weights_dir=str(weights_dir))
    model.load_state_dict(torch.load(checkpoint, map_location="cpu"))
    model.eval().to(DEVICE)

    loader = DataLoader(dataset, batch_size=8, shuffle=False,
                        num_workers=4, collate_fn=custom_collate)
    all_preds, all_labels = [], []
    with torch.no_grad():
        for X, labels, _ in loader:
            preds = torch.argmax(model(X.to(DEVICE)), dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    return {
        "f1":  round(f1_score(all_labels, all_preds, average="weighted"), 4),
        "acc": round(accuracy_score(all_labels, all_preds), 4),
        "mcc": round(matthews_corrcoef(all_labels, all_preds), 4),
        "n":   len(all_labels),
    }

CLASSIFIERS = [
    ("Lower bound",
     "/home/student/lisa_ma/results/VerSe_classifier/label_override_8ep_23092026_2135/spec.json"),
    ("Upper bound",
     "/home/student/lisa_ma/results/VerSe_classifier/upper_bound_8ep_25092026_1348/spec.json"),
]

# ── Run ───────────────────────────────────────────────────────────────────────
anomaly_dataset = AnomalyTestDataset(test_files, _transform, anomaly_lookup, lo_map)

rows = []
for name, spec_path in CLASSIFIERS:
    print(f"\nEvaluating {name}...")
    r = evaluate(name, spec_path, anomaly_dataset)
    rows.append({"Classifier": name, **r})
    print(f"  {name:22s}  F1={r['f1']:.4f}  Acc={r['acc']:.4f}  MCC={r['mcc']:.4f}  (n={r['n']})")

print("\n=== Summary (anomaly-corrected test set) ===")
print(f"{'Classifier':<22}  {'F1':>6}  {'Acc':>6}  {'MCC':>6}")
for r in rows:
    print(f"{r['Classifier']:<22}  {r['f1']:>6.4f}  {r['acc']:>6.4f}  {r['mcc']:>6.4f}")
