import numpy as np
import torch
import torch.nn as nn
from torch.func import functional_call, grad, vmap
import os
import re
import pandas as pd
import pytorch_lightning as pl
from tqdm import tqdm

from models.model_densenet import DenseNetModel  # DenseNet statt ResNet


pl.seed_everything(42)


def calculate_gradnorm(
    models,
    train_dataloader,
    device,
    num_classes
):
    
    scores = {
            "id": [],
            "epoch": [],
            "label": []
        }
    
    for c in range(num_classes):
        scores[f"c{c}_grad_norm"] = []

    CE = nn.CrossEntropyLoss(reduction="none").to(device)  
    
    for X, label, id in tqdm(train_dataloader, desc="Processing Samples", unit="sample"): 
        X, label = X.to(device), label.to(device) 

        for index, net in enumerate(models):
            net.eval()
            
            # Sammle alle Gradienten für alle Klassen für dieses Sample/Epoche
            batch_grads = {}
            
            for c in range(num_classes):
                net.zero_grad()  
                
                params = {k: v.detach() for k, v in net.named_parameters()}
                buffers = {k: v.detach() for k, v in net.named_buffers()}
                
                def compute_loss(params, buffers, sample, target):
                    sample = sample.unsqueeze(0)
                    target = target.unsqueeze(0)

                    logits = functional_call(net, (params, buffers), (sample,))
                    loss = CE(logits, target)
                    return loss[0]

                ft_compute_grad = grad(compute_loss)
                ft_compute_sample_grad = vmap(ft_compute_grad, in_dims=(None, None, 0, 0))
                c_tensor = torch.full_like(label, c).to(device)
                grads = ft_compute_sample_grad(params, buffers, X, c_tensor)
                del ft_compute_sample_grad

                # Berechne Gradienten-Normen für alle Samples in diesem Batch
                batch_grad_norms = []
                for i in range(len(X)):
                    grad_i = [grads[name][i].flatten() for name in grads]
                    flattened_grad = torch.cat(grad_i)
                    grad_norm = torch.linalg.norm(flattened_grad.flatten())
                    batch_grad_norms.append(float(grad_norm.detach().cpu().item()))
                
                batch_grads[c] = batch_grad_norms
            
            # Jetzt füge für jedes Sample alle Informationen gleichzeitig hinzu
            for i in range(len(X)):
                # Stelle sicher, dass ID ein String ist, nicht ein Tensor
                sample_id = str(id[i]) if not isinstance(id[i], str) else id[i]
                
                scores["id"].append(sample_id)
                scores["epoch"].append(index)
                scores["label"].append(int(label[i].detach().cpu().item()))
                
                # Füge alle Klassen-Gradienten für dieses Sample hinzu
                for c in range(num_classes):
                    scores[f"c{c}_grad_norm"].append(batch_grads[c][i])

    scores = pd.DataFrame(scores)
    
    # Debug: Prüfe auf doppelte ID-Epoch-Kombinationen
    duplicates = scores.groupby(['id', 'epoch']).size()
    duplicate_entries = duplicates[duplicates > 1]
    if len(duplicate_entries) > 0:
        print(f"⚠️  WARNUNG: {len(duplicate_entries)} doppelte ID-Epoch-Kombinationen gefunden!")
        print("Erste 10 doppelte Einträge:")
        print(duplicate_entries.head(10))
        
        # Entferne Duplikate
        print("Entferne Duplikate...")
        scores = scores.drop_duplicates(subset=['id', 'epoch'], keep='first')
        print(f"Nach Duplikat-Entfernung: {scores.shape}")
    
    return scores



def calculate_glare(grad_norms):
    print(f"calculate_glare: Eingabe DataFrame shape: {grad_norms.shape}")
    
    # Robuste Alternative: Verwende pivot_table statt pivot
    n_classes = grad_norms['label'].max() + 1
    class_indices = range(n_classes)
    gradient_columns = [f"c{c}_grad_norm" for c in class_indices]

    class_matrices = {}

    for col in gradient_columns:
        print(f"Verarbeite Spalte: {col}")
        
        # Verwende pivot_table mit aggfunc='first' um Duplikate zu behandeln
        matrix = grad_norms.pivot_table(index="id", columns="epoch", values=col, aggfunc='first')
        matrix = matrix.reset_index()
        matrix['class'] = col
        matrix = matrix.merge(grad_norms[['id', 'label']].drop_duplicates(), on='id', how='left')
        class_matrices[col] = matrix

    print(f"Alle {len(class_matrices)} Klassen-Matrizen erstellt")
    grad_norms_per_class = pd.concat(class_matrices.values(), ignore_index=True)
    print(f"Kombiniertes DataFrame shape: {grad_norms_per_class.shape}")

    n_epoch = grad_norms["epoch"].max() + 1
    scores_list = []

    def max_index_excluding_class_x(lst, x):
        filtered_list = np.delete(lst, x)
        max_value = max(filtered_list)
        max_index = np.where(lst==max_value)[0][0]
        return max_index

    print(f"Berechne GLARE-Scores für {len(grad_norms_per_class.groupby('id'))} Samples...")
    
    for sample_idx, (sample_id, group_df) in enumerate(grad_norms_per_class.groupby("id")):
        if sample_idx % 1000 == 0:
            print(f"  Verarbeitet: {sample_idx} Samples")

        glare = np.zeros(n_classes)
        min_class_sequence = []
        label = int(group_df["label"].values[0])
        grad_norm_in_min_epochs = np.zeros(n_classes)
        grad_norm_total = np.zeros(n_classes)

        # Get the lowest grad norm class for each epoch
        for epoch in range(n_epoch):
            if epoch in group_df.columns:
                values = group_df[epoch]
                # Filtere NaN-Werte heraus
                valid_values = values.dropna()
                if len(valid_values) > 0:
                    min_class = np.argmin(valid_values)
                    glare[min_class] += 1
                    grad_norm_in_min_epochs[min_class] += valid_values.values[min_class]
                    min_class_sequence.append(min_class)
                    grad_norm_total += valid_values.values
        
        if len(min_class_sequence) > 0:  # Nur wenn wir gültige Daten haben
            alt_class_glare = max_index_excluding_class_x(glare, label)
            
            # Compute avg lowest value for each class (only where it was the minimum)
            glarex = np.zeros(n_classes)

            for c in range(n_classes):
                if glare[c] > 0:
                    avg = grad_norm_in_min_epochs[c] / glare[c]
                    inv_avg = 1.0 / (avg) if avg > 0 else 0.0
                    glarex[c] = glare[c] + 0.01 * inv_avg
                else:
                    avg = grad_norm_total[c] / n_epoch if n_epoch > 0 else 0.0
                    inv_avg = 1.0 / (avg) if avg > 0 else 0.0
                    glarex[c] = glare[c] + 0.01 * inv_avg

            alt_class_glarex = max_index_excluding_class_x(glarex, label)
            
            scores_list.append([
                sample_id, 
                label,
                int(glare[label]), 
                float(glarex[label]), 
                alt_class_glare,
                int(glare[int(alt_class_glare)]),
                alt_class_glarex,
                float(glarex[int(alt_class_glarex)])])

    print(f"GLARE-Berechnung abgeschlossen: {len(scores_list)} gültige Samples")

    scores_df = pd.DataFrame(scores_list, columns=[
        "id", 
        "label",  # Zurück zu "label" für Konsistenz mit Original
        "glare score", 
        "glarex score",
        "alternative class (glare)",
        "alternative glare score",
        "alternative class (glarex)",
        "alternative glarex score"])
    
    return scores_df


def compute_glare_densenet(dataloader, num_classes, weights_dir, spec):
    """
    GLARE-Analyse speziell für DenseNet-Modelle
    """
    pattern = re.compile(r'epoch_\d+$')  

    weights = [
        os.path.join(weights_dir, file)
        for file in os.listdir(weights_dir)
        if pattern.match(file)
    ]
    weights.sort(key=lambda x: int(x.rsplit('_', 1)[-1]))
    
    print(f"Gefundene Gewichte: {len(weights)} Epochen")
    for w in weights:
        print(f"  - {os.path.basename(w)}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Verwende Device: {device}")

    # Create DenseNet models for each epoch
    models = [] 
    for index, weight_path in enumerate(weights):
        print(f"Lade Modell für Epoche {index}...")
        
        # Erstelle DenseNet-Modell
        full_model = DenseNetModel(
            num_classes=num_classes,
            spec=spec,
            weights_dir=weights_dir
        )
        
        # Lade State Dict
        state_dict = torch.load(weight_path, map_location=device)
        full_model.load_state_dict(state_dict)
        
        # Extrahiere nur das DenseNet (ohne Lightning-Wrapper)
        net = full_model.model.to(device)
        net.eval()
        
        models.append(net)

    print(f"Alle {len(models)} Modelle geladen!")

    grad_norms = calculate_gradnorm(models=models, train_dataloader=dataloader, device=device, num_classes=num_classes)
    scores = calculate_glare(grad_norms)
    
    return scores, grad_norms


# Wrapper-Funktion für Kompatibilität
def compute_glare(dataloader, num_classes, weights_dir, spec):
    """
    Automatische Erkennung und Verwendung der richtigen GLARE-Funktion
    """
    model_type = spec.get('model', 'unknown')
    
    if 'densenet' in model_type.lower():
        print(f"Erkannter Modelltyp: DenseNet -> verwende compute_glare_densenet")
        return compute_glare_densenet(dataloader, num_classes, weights_dir, spec)
    else:
        print(f"Unbekannter Modelltyp: {model_type} -> verwende Standard-GLARE (möglicherweise inkompatibel)")
        # Hier könnte die originale compute_glare Funktion aufgerufen werden
        raise NotImplementedError(f"GLARE für Modelltyp '{model_type}' noch nicht implementiert")