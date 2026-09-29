"""
Run forward-pass inference on the training set and save per-sample softmax
probabilities. No gradient computation — much faster than GLARE.

Used to compute patient-level T13 suspicion scores:
  For each patient, look at the Thoracic probability for Lumbar-labeled
  vertebrae. A T13 patient should show elevated p_Thoracic at vert20
  even when integer GLARE score = 8.
"""

import json
import os
import re
import torch
import torch.nn as nn
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from datetime import datetime
from monai.transforms import Compose, NormalizeIntensity

from models.model_densenet import DenseNetModel
from dataloaders.dataloader_VerSe_new import VerSeDataLoader
from helper.arguments import get_parameter

CLASS_NAMES = ["Cervical", "Thoracic", "Lumbar", "Sacral"]

# ── Config ────────────────────────────────────────────────────────────────────
SPEC_PATH   = "results/VerSe_classifier/label_override_8ep_23092026_2135/spec.json"
WEIGHTS_DIR = "results/VerSe_classifier/label_override_8ep_23092026_2135/weights"
EPOCH_FILE  = os.path.join(WEIGHTS_DIR, "epoch_7")   # last epoch

SPEC_DATA   = {"mean": [-111.3885], "std": [406.3665]}
BATCH_SIZE  = 4   # larger than GLARE's 1 — no gradient needed, faster
NUM_WORKERS = 6

# ── Setup ─────────────────────────────────────────────────────────────────────
class ImageTransform:
    def __init__(self, mean, std):
        self.transforms = Compose([
            NormalizeIntensity(subtrahend=mean, divisor=std, channel_wise=True),
        ])
    def __call__(self, data):
        return self.transforms(data)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

with open(SPEC_PATH) as f:
    spec = json.load(f)

spec["dataset_dir"] = "/home/student/lisa_ma/prepared"
spec["batch_size"]  = BATCH_SIZE

# ── Load model (last epoch only) ──────────────────────────────────────────────
print(f"Loading weights from {EPOCH_FILE}")
full_model = DenseNetModel(num_classes=4, spec=spec, weights_dir=WEIGHTS_DIR)
state_dict = torch.load(EPOCH_FILE, map_location=device)
full_model.load_state_dict(state_dict)
net = full_model.model.to(device)
net.eval()

softmax_fn = nn.Softmax(dim=1)

# ── Dataloader ────────────────────────────────────────────────────────────────
data_module = VerSeDataLoader(
    spec=spec,
    train_transforms=ImageTransform(mean=SPEC_DATA["mean"], std=SPEC_DATA["std"]),
    val_transforms=ImageTransform(mean=SPEC_DATA["mean"], std=SPEC_DATA["std"]),
    num_workers=NUM_WORKERS,
)
data_module.setup()
train_loader = data_module.train_dataloader()
print(f"Running inference on {len(train_loader.dataset)} samples...\n")

# ── Inference ─────────────────────────────────────────────────────────────────
rows = []
with torch.no_grad():
    for X, label, ids in tqdm(train_loader, desc="Inference", unit="batch"):
        X      = X.to(device)
        logits = net(X)
        probs  = softmax_fn(logits).cpu().numpy()

        for i, sample_id in enumerate(ids):
            lbl = int(label[i].item())
            rows.append({
                "sample_id":  sample_id,
                "label":      lbl,
                "label_name": CLASS_NAMES[lbl],
                "p_Cervical": float(probs[i, 0]),
                "p_Thoracic": float(probs[i, 1]),
                "p_Lumbar":   float(probs[i, 2]),
                "p_Sacral":   float(probs[i, 3]),
                "pred_class": int(probs[i].argmax()),
                "pred_name":  CLASS_NAMES[int(probs[i].argmax())],
            })

df = pd.DataFrame(rows)

# ── Save ──────────────────────────────────────────────────────────────────────
out_dir  = Path("results/inference_softmax")
out_dir.mkdir(parents=True, exist_ok=True)
ts       = datetime.now().strftime("%Y%m%d_%H%M%S")
out_path = out_dir / f"softmax_probs_{ts}.csv"
df.to_csv(out_path, index=False)
print(f"\nSaved {len(df)} rows → {out_path}")

# ── Quick sanity check ────────────────────────────────────────────────────────
correct = (df["label"] == df["pred_class"]).mean()
print(f"Overall prediction accuracy: {correct:.2%}")
print("\nPrediction accuracy per class:")
for cls_int, cls_name in enumerate(CLASS_NAMES):
    subset = df[df["label"] == cls_int]
    if len(subset):
        acc = (subset["pred_class"] == cls_int).mean()
        print(f"  {cls_name:10s} (n={len(subset):5d}): {acc:.2%}")

# ── Patient-level T13 suspicion score ─────────────────────────────────────────
# For each patient: max p_Thoracic among all Lumbar-labeled vertebrae.
# T13 patients should have vert20 with elevated p_Thoracic.
print("\n── Patient-level T13 suspicion (max p_Thoracic in Lumbar-labeled verts) ──")

GT_DERIVED = Path("evaluation/gt_mislabels_derived.csv")
gt = pd.read_csv(GT_DERIVED)
t13_pids = set(gt[(gt["anomaly_type"] == "T13_extra") & (gt["vert_id"] == 20)]["pid"].astype(str))

# Extract pid (everything before _vert\d+$)
df["pid"]     = df["sample_id"].str.replace(r"_vert\d+$", "", regex=True)
df["vert_id"] = df["sample_id"].str.extract(r"_vert(\d+)$").astype(float)

lumbar_df = df[df["label"] == 2].copy()   # Lumbar-labeled only
pat_score = (
    lumbar_df
    .groupby("pid")["p_Thoracic"]
    .max()
    .rename("max_p_Thoracic_in_Lumbar")
    .reset_index()
)
pat_score["is_t13"] = pat_score["pid"].isin(t13_pids)

t13_scores   = pat_score[pat_score["is_t13"]]["max_p_Thoracic_in_Lumbar"]
clean_scores = pat_score[~pat_score["is_t13"]]["max_p_Thoracic_in_Lumbar"]

print(f"\nT13 patients  (n={len(t13_scores)}):  "
      f"mean={t13_scores.mean():.4f}  median={t13_scores.median():.4f}  "
      f"min={t13_scores.min():.4f}  max={t13_scores.max():.4f}")
print(f"Clean patients (n={len(clean_scores)}): "
      f"mean={clean_scores.mean():.4f}  median={clean_scores.median():.4f}  "
      f"min={clean_scores.min():.4f}  max={clean_scores.max():.4f}")

# Distribution of T13 scores vs clean at different thresholds
print("\nAt threshold p_Thoracic ≥ X, how many T13 patients flagged vs FP rate:")
for thresh in [0.05, 0.10, 0.20, 0.30, 0.50]:
    t13_flagged   = (t13_scores   >= thresh).sum()
    clean_flagged = (clean_scores >= thresh).sum()
    fpr = clean_flagged / len(clean_scores)
    print(f"  ≥{thresh:.2f}: T13 flagged {t13_flagged:2d}/{len(t13_scores)}  |  "
          f"clean flagged {clean_flagged:4d}/{len(clean_scores)} (FPR {fpr:.1%})")

print(f"\nResults saved to: {out_path}")
