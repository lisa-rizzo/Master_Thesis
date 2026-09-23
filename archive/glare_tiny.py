from glare import compute_glare
from dataloaders.dataloader_VerSe import VerSeDataLoader
from helper.arguments import get_parameter
from monai.transforms import Compose, NormalizeIntensity
import json
import os
import shutil
import re
import torch
from torch.utils.data import Subset

# Für absoluten Mini-Test: Nur 10 Samples, 2 Epochen
class ImageTransform:
    def __init__(self, mean, std):
        self.transforms = Compose([
            NormalizeIntensity(subtrahend=mean, divisor=std, channel_wise=True),
        ])
    def __call__(self, data):
        return self.transforms(data)

print("=== ULTRA-MINI DEBUG TEST ===")

# Spec laden
with open("/home/student/lisa_ma/results/VerSe_classifier/test_glare_27092025_1811/spec.json", "r") as f:
    spec = json.load(f)

spec_data = {"mean": [-111.3885], "std": [406.3665]}
spec["dataset_dir"] = "/home/student/lisa_ma/prepared"
spec["train_set_size"] = 1.0  # Alle Daten laden, aber dann nur x nehmen
spec["batch_size"] = 4
weights_dir = "/home/student/lisa_ma/results/VerSe_classifier/test_glare_27092025_1811/weights"

# Nur 2 Epochen kopieren
all_files = os.listdir(weights_dir)
epoch_pattern = re.compile(r'^epoch_(\d+)$')
epoch_numbers = [int(re.match(r'^epoch_(\d+)$', f).group(1)) for f in all_files if re.match(r'^epoch_(\d+)$', f)]
epoch_numbers.sort()
last_2_epochs = epoch_numbers[-3:]

temp_weights_dir = f"{weights_dir}_debug"
if os.path.exists(temp_weights_dir):
    shutil.rmtree(temp_weights_dir)
os.makedirs(temp_weights_dir)

for epoch_num in last_2_epochs:
    src = f"{weights_dir}/epoch_{epoch_num}"
    dst = f"{temp_weights_dir}/epoch_{epoch_num}"
    shutil.copy2(src, dst)

# DataLoader mit nur 10 Samples
data_module = VerSeDataLoader(
    spec=spec,
    train_transforms=ImageTransform(mean=spec_data["mean"], std=spec_data["std"]),
    val_transforms=ImageTransform(mean=spec_data["mean"], std=spec_data["std"]),
    num_workers=1
)
data_module.setup()
full_dataset = data_module.train_dataset

# Nur die ersten 10 Samples nehmen
tiny_dataset = Subset(full_dataset, range(1000))
train_loader = torch.utils.data.DataLoader(
    tiny_dataset, 
    batch_size=4, 
    shuffle=False,
    collate_fn=data_module.train_dataloader().collate_fn
)

print(f"DEBUG: Teste mit nur {len(tiny_dataset)} Samples, {len(last_2_epochs)} Epochen")

try:
    scores, grad_norms = compute_glare(
        dataloader=train_loader,
        num_classes=4,
        weights_dir=temp_weights_dir,
        spec=spec
    )

# MISLABEL-IDENTIFIKATION hinzufügen
    method = get_parameter(spec, "method", "glarex", str)
    threshold_fraction = get_parameter(spec, "threshold_fraction", 0.1, float)
    threshold = scores[f'{method} score'].quantile(threshold_fraction)
    
    # Mislabels identifizieren (niedrigste X% der Scores)
    mislabels = scores[scores[f'{method} score'] <= threshold]
    scores['is_mislabel'] = scores[f'{method} score'] <= threshold
    
    print(f"\n🎉 DEBUG SUCCESS!")
    print(f"Analysierte Samples: {len(scores)}")
    print(f"Grad-norms: {len(grad_norms)} rows")
    print(f"Threshold ({threshold_fraction*100:.0f}%): {threshold:.6f}")
    print(f"Identifizierte Mislabels: {len(mislabels)} von {len(scores)} ({len(mislabels)/len(scores)*100:.1f}%)")
    
    if len(mislabels) > 0:
        print(f"\n🚨 TOP 10 MISLABELS:")
        print("Rang  GLAREX Score  ID                           Label  Alternative")
        print("-" * 75)
        mislabels_sorted = mislabels.sort_values(f'{method} score')
        for i, (idx, row) in enumerate(mislabels_sorted.head(10).iterrows(), 1):
            print(f"{i:4d}  {row[f'{method} score']:11.6f}  {str(row['id'])[:28]:28s} {row['label']:5d}  {row[f'alternative class ({method})']:11d}")
    else:
        print(f"\n✅ Keine Mislabels in diesem Sample gefunden")
    
    print(f"\n📊 VERTEILUNG DER KLASSEN:")
    class_counts = scores['label'].value_counts().sort_index()
    for class_idx, count in class_counts.items():
        class_names = {0: "Cervical", 1: "Thoracic", 2: "Lumbar", 3: "Sacral"}
        class_name = class_names.get(class_idx, f"Class {class_idx}")
        print(f"  Klasse {class_idx} ({class_name:8s}): {count:3d} Samples")
    
    print(f"\n📈 GLARE SCORE STATISTIKEN:")
    print(f"  Min:    {scores[f'{method} score'].min():.6f}")
    print(f"  Max:    {scores[f'{method} score'].max():.6f}")
    print(f"  Mean:   {scores[f'{method} score'].mean():.6f}")
    print(f"  Median: {scores[f'{method} score'].median():.6f}")
    
    # Ergebnisse speichern
    scores.to_csv('mini_test_scores.csv', index=False)
    mislabels.to_csv('mini_test_mislabels.csv', index=False)
    
    print(f"\n💾 GESPEICHERT:")
    print(f"  mini_test_scores.csv - Alle {len(scores)} Samples mit GLARE-Scores")
    print(f"  mini_test_mislabels.csv - {len(mislabels)} identifizierte Mislabels")
    
    
    print(f"\n🎉 DEBUG SUCCESS!")
    print(f"Scores: {len(scores)} rows")
    print(f"Grad-norms: {len(grad_norms)} rows") 
    print("\nErste Scores:")
    print(scores)
    
except Exception as e:
    print(f"\n❌ DEBUG FEHLER: {e}")
    import traceback
    traceback.print_exc()

finally:
    shutil.rmtree(temp_weights_dir)