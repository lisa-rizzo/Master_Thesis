"""Quick evaluation of lower bound and upper bound classifiers on the test set."""
import sys
sys.path.insert(0, "/home/student/lisa_ma")

import json
import torch
import numpy as np
from pathlib import Path
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, matthews_corrcoef, accuracy_score
from monai.transforms import Compose, NormalizeIntensity

from dataloaders.dataloader_VerSe import VerSeDataLoader, VerSeDataset, custom_collate
import datasets.VerSe.get_data_VerSe as get_data_VerSe
from models.model_densenet import DenseNetModel

DATASET_DIR = "/home/student/lisa_ma/prepared"
EXCEL_PATH  = "/home/student/lisa_ma/datasets/VerSe/data_filter_joined.xlsx"
SPEC_DATA   = {"mean": [-111.3885], "std": [406.3665]}
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")

class Normalize:
    def __init__(self):
        self.t = Compose([NormalizeIntensity(
            subtrahend=SPEC_DATA["mean"], divisor=SPEC_DATA["std"], channel_wise=True
        )])
    def __call__(self, x):
        return self.t(x)

_transform = Normalize()

# Load pid_corrections from lower bound spec
LB_SPEC_PATH = "/home/student/lisa_ma/results/VerSe_classifier/label_override_8ep_23092026_2135/spec.json"
with open(LB_SPEC_PATH) as f:
    _lb_spec = json.load(f)
_lb_spec["dataset_dir"] = DATASET_DIR
_lb_spec["batch_size"] = 1
_dm = VerSeDataLoader(spec=_lb_spec, train_transforms=_transform, val_transforms=_transform, num_workers=0)
_dm.setup()
PID_CORRECTIONS = _dm.train_dataset.pid_corrections
print(f"pid_corrections loaded: {len(PID_CORRECTIONS)} PIDs")

# Load test files
TEST_FILES, _ = get_data_VerSe.get_filtered_files_across_dsnames(
    root_dir=DATASET_DIR,
    excel_path=EXCEL_PATH,
    split="test",
    glob_pattern="*.npz",
    pid_col="pid",
    dsname_col="dsname",
    verts_col="vert_label",
    check_complete=True,
    apply_excel_filter=True,
    extra_sacral_verts=frozenset(),
)
print(f"Test samples: {len(TEST_FILES)}\n")

def evaluate(name, spec_path, epoch=7):
    spec_path = Path(spec_path)
    with open(spec_path) as f:
        spec = json.load(f)
    weights_dir = spec_path.parent / "weights"
    checkpoint  = weights_dir / f"epoch_{epoch}"

    test_dataset = VerSeDataset(TEST_FILES, transform=_transform)
    test_dataset.pid_corrections = PID_CORRECTIONS
    test_loader = DataLoader(
        test_dataset, batch_size=8, shuffle=False, num_workers=4, collate_fn=custom_collate
    )

    model = DenseNetModel(num_classes=4, spec=spec, weights_dir=str(weights_dir))
    model.load_state_dict(torch.load(checkpoint, map_location="cpu"))
    model.eval().to(DEVICE)

    all_preds, all_labels = [], []
    with torch.no_grad():
        for X, labels, _ in test_loader:
            preds = torch.argmax(model(X.to(DEVICE)), dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    f1  = f1_score(all_labels, all_preds, average="weighted")
    mcc = matthews_corrcoef(all_labels, all_preds)
    acc = accuracy_score(all_labels, all_preds)
    print(f"  {name:22s}  F1={f1:.4f}  Acc={acc:.4f}  MCC={mcc:.4f}  (n={len(all_labels)})")
    return {"name": name, "f1": f1, "acc": acc, "mcc": mcc}

CLASSIFIERS = [
    ("Lower bound",
     "/home/student/lisa_ma/results/VerSe_classifier/label_override_8ep_23092026_2135/spec.json"),
    ("Upper bound",
     "/home/student/lisa_ma/results/VerSe_classifier/upper_bound_8ep_25092026_1348/spec.json"),
]

results = []
for name, path in CLASSIFIERS:
    print(f"Evaluating {name}...")
    results.append(evaluate(name, path))

print("\n=== Summary ===")
print(f"{'Classifier':<22}  {'F1':>6}  {'Acc':>6}  {'MCC':>6}")
for r in results:
    print(f"{r['name']:<22}  {r['f1']:>6.4f}  {r['acc']:>6.4f}  {r['mcc']:>6.4f}")
