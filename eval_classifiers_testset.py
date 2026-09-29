"""
eval_classifiers_testset.py — Test-set performance of all four classifiers.

Evaluated at the last saved epoch (epoch_7) on the anomaly-corrected test set.
Ground truth corrections applied (T13/T11/LO) from A_CT_ANOMALY_LABELv9.xlsx.
Results printed as a table and saved to evaluation/classifier_testset_results.csv.

Usage:
  python eval_classifiers_testset.py
  python eval_classifiers_testset.py --epoch 5   # evaluate a different epoch
"""
import sys, argparse, json
sys.path.insert(0, "/home/student/lisa_ma")

import numpy as np
import pandas as pd
import torch
from pathlib import Path
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, matthews_corrcoef, accuracy_score
from monai.transforms import Compose, NormalizeIntensity

import datasets.VerSe.get_data_VerSe_new as get_data_VerSe
from dataloaders.dataloader_VerSe import VerSeDataset, custom_collate
from models.model_densenet import DenseNetModel

# ── Config ─────────────────────────────────────────────────────────────────────
DATASET_DIR   = "/home/student/lisa_ma/prepared"
FILTER_EXCEL  = "/home/student/lisa_ma/datasets/VerSe/data_filter_joined.xlsx"
ANOMALY_EXCEL = "/home/student/lisa_ma/evaluation/A_CT_ANOMALY_LABELv9.xlsx"
OUT_CSV       = "/home/student/lisa_ma/evaluation/classifier_testset_results.csv"
SPEC_DATA     = {"mean": [-111.3885], "std": [406.3665]}
DEVICE        = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CLASSIFIERS = [
    ("Lower bound",
     "/home/student/lisa_ma/results/VerSe_classifier/label_override_8ep_23092026_2135/spec.json"),
    ("Corrected mislabels (GLAREx 0.2%)",
     "/home/student/lisa_ma/results/VerSe_classifier/glare_correction_frac02_25092026_1959/spec.json"),
    ("Corrected mislabels (GLAREx 0.4%)",
     "/home/student/lisa_ma/results/VerSe_classifier/glare_correction_frac04_26092026_0122/spec.json"),
    ("Upper bound",
     "/home/student/lisa_ma/results/VerSe_classifier/upper_bound_8ep_25092026_1348/spec.json"),
]

# ── Region mapping (vert 28 = T13 = Thoracic) ─────────────────────────────────
def get_region_label(vert: int) -> int:
    if 1 <= vert <= 7:                  return 0
    if (8 <= vert <= 19) or vert == 28: return 1
    if 20 <= vert <= 24:                return 2
    if vert >= 25:                      return 3
    return -1

# ── Dataset with anomaly corrections ──────────────────────────────────────────
class AnomalyTestDataset(VerSeDataset):
    def __init__(self, file_paths, transform, anomaly_lookup, lo_map):
        super().__init__(file_paths, transform)
        self._anomaly = anomaly_lookup
        self._lo      = lo_map

    def __getitem__(self, idx):
        path = self.file_paths[idx]
        img  = torch.tensor(np.load(path)["img"], dtype=torch.float32).unsqueeze(0)
        if self.transform:
            img = self.transform(img)

        disk_vert = self.get_vert_number(path)
        pid       = self.get_pid_from_path(path)
        corrected = disk_vert

        if pid in self._anomaly:
            info = self._anomaly[pid]
            V    = info["vert"]
            if info["T13"]:
                if disk_vert == V:    corrected = 28
                elif disk_vert > V:   corrected = disk_vert - 1
            elif info["T11"]:
                if disk_vert >= V:    corrected = disk_vert + 1

        if pid in self._lo:
            corrected = self._lo[pid].get(disk_vert, disk_vert)

        return img, get_region_label(corrected), f"{pid}_vert{corrected}"


def build_anomaly_lookup(path: str) -> dict:
    df = pd.read_excel(path)
    if "Remove" in df.columns:
        df = df[df["Remove"] != 1]
    lookup = {}
    for _, row in df.iterrows():
        fid = row.get("fid"); ds = row.get("dataset")
        if pd.isna(fid) or pd.isna(ds):
            continue
        t13 = row.get("T13", 0); is_T13 = not pd.isna(t13) and int(t13) == 1
        t11 = row.get("T11", 0); is_T11 = not pd.isna(t11) and int(t11) == 1
        raw_v = row.get("vert")
        vert  = int(raw_v) if raw_v is not None and not pd.isna(raw_v) else None
        if not (is_T13 or is_T11) or vert is None:
            continue
        lookup[str(fid)] = {"T13": is_T13, "T11": is_T11, "vert": vert}
    return lookup


def evaluate(name: str, spec_path: str, dataset: AnomalyTestDataset, epoch: int) -> dict:
    spec_path = Path(spec_path)
    with open(spec_path) as f:
        spec = json.load(f)
    checkpoint = spec_path.parent / "weights" / f"epoch_{epoch}"

    model = DenseNetModel(num_classes=4, spec=spec,
                          weights_dir=str(spec_path.parent / "weights"))
    model.load_state_dict(torch.load(checkpoint, map_location="cpu"))
    model.eval().to(DEVICE)

    loader = DataLoader(dataset, batch_size=8, shuffle=False,
                        num_workers=4, collate_fn=custom_collate)
    preds, labels = [], []
    with torch.no_grad():
        for X, y, _ in loader:
            preds.extend(torch.argmax(model(X.to(DEVICE)), dim=1).cpu().numpy())
            labels.extend(y.cpu().numpy())

    return {
        "classifier": name,
        "epoch":      epoch,
        "n":          len(labels),
        "acc":        round(accuracy_score(labels, preds), 4),
        "f1":         round(f1_score(labels, preds, average="weighted"), 4),
        "mcc":        round(matthews_corrcoef(labels, preds), 4),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--epoch", type=int, default=7)
    parser.add_argument("--out", default=OUT_CSV)
    args = parser.parse_args()

    print(f"Device: {DEVICE}  |  Epoch: {args.epoch}\n")

    transform = Compose([NormalizeIntensity(
        subtrahend=SPEC_DATA["mean"], divisor=SPEC_DATA["std"], channel_wise=True
    )])

    test_files, _ = get_data_VerSe.get_filtered_files_across_dsnames(
        root_dir=DATASET_DIR, excel_path=FILTER_EXCEL, split="test",
        glob_pattern="*.npz", pid_col="pid", dsname_col="dsname",
        verts_col="vert_label", check_complete=True, apply_excel_filter=True,
        extra_sacral_verts=frozenset(),
    )
    print(f"Test samples: {len(test_files)}")

    anomaly_lookup = build_anomaly_lookup(ANOMALY_EXCEL)
    lo_map         = get_data_VerSe.load_label_override_step1(FILTER_EXCEL, ANOMALY_EXCEL)

    dataset = AnomalyTestDataset(test_files, transform, anomaly_lookup, lo_map)

    rows = []
    for name, spec_path in CLASSIFIERS:
        r = evaluate(name, spec_path, dataset, args.epoch)
        rows.append(r)

    df = pd.DataFrame(rows)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f"\nResults saved to: {args.out}")

    print(f"\n{'Dataset':<25}  {'Acc':>6}  {'F1':>6}  {'MCC':>6}")
    print("-" * 50)
    for r in rows:
        print(f"{r['classifier']:<25}  {r['acc']:>6.4f}  {r['f1']:>6.4f}  {r['mcc']:>6.4f}")


if __name__ == "__main__":
    main()
