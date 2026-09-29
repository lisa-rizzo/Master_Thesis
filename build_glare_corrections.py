"""
Build correction Excel files for fraction-based thresholds.

Sorts all samples by glarex score ascending (most suspicious first) and takes the
bottom FRACTION of training data. Outputs one xlsx per fraction to evaluation/.
Format required by the dataloader gt_corrections_path mechanism:
  columns: sample_id, true_class (one of Cervical/Thoracic/Lumbar/Sacral)
"""

import numpy as np
import pandas as pd
from pathlib import Path

GLARE_CSV = Path(
    "results/glare/19-08-26_21:11_ep-8_single-gpu/results/"
    "mislabels_list_20260819_211102.csv"
)
OUT_DIR = Path("evaluation")
INT_TO_CLASS = {0: "Cervical", 1: "Thoracic", 2: "Lumbar", 3: "Sacral"}

df = pd.read_csv(GLARE_CSV)
df_sorted = df.sort_values("glarex score", ascending=True).reset_index(drop=True)
n_total = len(df_sorted)
print(f"Loaded {n_total} samples from GLARE CSV.")

for frac in [0.002, 0.004]:
    k = max(1, int(np.ceil(frac * n_total)))
    flagged = df_sorted.head(k)
    corrections = pd.DataFrame({
        "sample_id": flagged["id"].values,
        "true_class": flagged["alternative class (glarex)"].map(INT_TO_CLASS).values,
    })
    unmapped = corrections["true_class"].isna().sum()
    if unmapped:
        print(f"  Warning: {unmapped} rows with unmapped alternative class at fraction {frac:.1%}")
        corrections = corrections.dropna(subset=["true_class"])

    frac_label = f"{frac*100:.1f}".replace(".", "")  # "002" -> "02", "004" -> "04"
    out_path = OUT_DIR / f"glare_corrections_frac{frac_label}.xlsx"
    corrections.to_excel(out_path, index=False)
    print(f"Fraction {frac:.1%} ({k} samples) → {out_path}")

print("Done.")
