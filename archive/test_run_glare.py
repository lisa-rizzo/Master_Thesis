import json
import os
import shutil
import torch
from glare_densenet import compute_glare  # Verwende DenseNet-kompatible Version                
from dataloaders.dataloader_VerSe import VerSeDataLoader
from helper.arguments import get_parameter    
import numpy as np
from sklearn.metrics import accuracy_score, recall_score, f1_score, matthews_corrcoef
import torch.nn as nn
from models.model_densenet import DenseNetModel  # DenseNet statt ResNet    

# GPU Memory komplett freigeben
torch.cuda.empty_cache()
torch.cuda.synchronize()

# 2. Spezifikation und Parameter laden (wie im Training)
with open("/home/student/lisa_ma/results/VerSe_classifier/test_glare_27092025_1811/spec.json", "r") as f:
    spec = json.load(f)

# Zeige geladene Spezifikation
print(f"Geladene Spezifikation:")
print(f"- Modell: {spec.get('model', 'unbekannt')}")
print(f"- Epochen trainiert: {spec.get('epochs', 'unbekannt')}")
print(f"- Batch Size original: {spec.get('batch_size', 'unbekannt')}")

# Für GLARE-Analyse anpassen
spec["batch_size"] = 4  # Kleinere Batch Size für GLARE (Memory-intensiv)
spec["train_set_size"] = 1.0  # ALLE Daten für komplette Mislabel-Analyse verwenden

# GLARE-Parameter sind bereits in spec.json vorhanden!
print(f"\nGLARE-Parameter aus Training:")
print(f"- Methode: {spec.get('method', 'nicht gesetzt')}")
print(f"- Remove mislabels: {spec.get('remove_mislabels', 'nicht gesetzt')}")
print(f"- Threshold fraction: {spec.get('threshold_fraction', 'nicht gesetzt')}")
print(f"- Verwende {spec['train_set_size']*100:.0f}% der verfügbaren Daten")


num_classes = 4                               
weights_dir = "/home/student/lisa_ma/results/VerSe_classifier/test_glare_27092025_1811/weights"
print(f"Weights-Verzeichnis: {weights_dir}")

# Prüfe verfügbare Epochen und verwende nur die letzten 2
epoch_files = [f for f in os.listdir(weights_dir) if f.startswith('epoch_') and f[6:].isdigit()]
if epoch_files:
    epoch_numbers = [int(f.split('_')[1]) for f in epoch_files]
    epoch_numbers.sort()

    # Verwende nur die letzten 2 Epochen
    last_2_epochs = epoch_numbers[-2:] if len(epoch_numbers) >= 2 else epoch_numbers

    print(f"Verfügbare Epochen: {epoch_numbers}")
    print(f"Verwende die letzten {len(last_2_epochs)} Epochen: {last_2_epochs}")

    # Erstelle temporären Ordner nur mit den letzten 2 Epochen
    temp_weights_dir = f"{weights_dir}_temp_last2"
    os.makedirs(temp_weights_dir, exist_ok=True)

    # Kopiere nur die letzten 2 Epochen
    for epoch_num in last_2_epochs:
        epoch_file = f"epoch_{epoch_num}"
        if os.path.exists(f"{weights_dir}/{epoch_file}"):
            shutil.copy2(f"{weights_dir}/{epoch_file}", f"{temp_weights_dir}/{epoch_file}")
            print(f"  Kopiert: {epoch_file}")
    
    # Verwende temporären Ordner für GLARE
    weights_dir = temp_weights_dir
    print(f"GLARE wird mit {len(last_2_epochs)} Epochen durchgeführt (viel schneller!)")
    
else:
    raise FileNotFoundError(f"Keine Epochen-Dateien in {weights_dir} gefunden")

# 3. Dataloader initialisieren und Daten vorbereiten
data_module = VerSeDataLoader(spec=spec, num_workers=0)
data_module.setup()
train_dataloader = data_module.train_dataloader()  # Trainingsdaten laden

# 4. GLARE-Analyse ausführen - ANGEPASST FÜR DENSENET
print(f"\nStarte GLARE-Analyse...")
print(f"Modell-Typ: {spec.get('model', 'unbekannt')}")
print(f"Weights-Verzeichnis: {weights_dir}")
print(f"Analysiere {len(train_dataloader.dataset)} Samples...")

# Verwende DenseNet-kompatible GLARE-Analyse
scores, grad_norms = compute_glare(
    dataloader=train_dataloader,              # Trainingsdaten
    num_classes=num_classes,                  # Anzahl der Klassen
    weights_dir=weights_dir,                  # Verzeichnis mit Modellgewichten
    spec=spec                                # Spezifikation
)
print(f"GLARE-Analyse erfolgreich abgeschlossen!")

# 5. GLARE-basierte Mislabel-Analyse - VEREINFACHT für erste Liste
method = get_parameter(spec, "method", "glarex", str)
threshold_fraction = get_parameter(spec, "threshold_fraction", 0.1, float)

# Berechne Threshold
threshold = scores[f'{method} score'].quantile(threshold_fraction)
mislabels = scores[scores[f'{method} score'] <= threshold]

print(f"\n=== MISLABEL LISTE ===")
print(f"Methode: {method}")
print(f"Threshold: {threshold:.4f} ({threshold_fraction*100}% niedrigste Scores)")
print(f"Gefundene Mislabels: {len(mislabels)} von {len(scores)} Bildern ({len(mislabels)/len(scores)*100:.1f}%)")
print(f"\n=== LISTE ALLER MISLABELS ===")

# Sortiere nach niedrigstem Score (schlechteste zuerst)
mislabels_sorted = mislabels.sort_values(f'{method} score')

print(f"{'Rang':<5} {'GLARE Score':<12} {'Bild ID':<15} {'Echtes Label':<12} {'Vorschlag':<12}")
print("-" * 70)

for i, (idx, row) in enumerate(mislabels_sorted.iterrows(), 1):
    score = row[f'{method} score']
    img_id = row['id']
    true_class = row['true class']
    alt_class = row[f'alternative class ({method})']
    
    print(f"{i:<5} {score:<12.4f} {img_id:<15} {true_class:<12} {alt_class:<12}")

print(f"\n=== ZUSAMMENFASSUNG ===")
print(f"Schlechtester Score: {mislabels_sorted.iloc[0][f'{method} score']:.4f}")
print(f"Bester Score (der Mislabels): {mislabels_sorted.iloc[-1][f'{method} score']:.4f}")

# Zeige Label-Verteilung der Mislabels
print(f"\n=== LABEL-VERTEILUNG DER MISLABELS ===")
true_labels = mislabels_sorted['true class'].value_counts()
print("Echte Labels:")
for label, count in true_labels.items():
    print(f"  Klasse {label}: {count} Bilder")

suggested_labels = mislabels_sorted[f'alternative class ({method})'].value_counts()
print("\nVorgeschlagene Labels:")
for label, count in suggested_labels.items():
    print(f"  Klasse {label}: {count} Bilder")

# 6. Ergebnisse speichern
scores.to_pickle(f'{weights_dir}/scores.pkl')
grad_norms.to_pickle(f'{weights_dir}/grad_norms.pkl')

# Speichere Mislabel-Liste als separate CSV
mislabels_list = mislabels_sorted[['id', f'{method} score', 'true class', f'alternative class ({method})']]
mislabels_list.to_csv(f'{weights_dir}/mislabels_list.csv', index=False)

print(f"\n=== DATEIEN GESPEICHERT ===")
print(f"- Mislabel-Liste: {weights_dir}/mislabels_list.csv")
print(f"- Vollständige Scores: {weights_dir}/scores.pkl")
print(f"- Gradient Norms: {weights_dir}/grad_norms.pkl")

# 7. Cleanup: Temporären Ordner löschen
if 'temp_weights_dir' in locals() and os.path.exists(temp_weights_dir):
    shutil.rmtree(temp_weights_dir)
    print(f"\nTemporärer Ordner {temp_weights_dir} wurde gelöscht")

print(f"\n=== MISLABEL-ANALYSE ABGESCHLOSSEN ===")
print(f"Die Liste der {len(mislabels)} Mislabels wurde erstellt!")

