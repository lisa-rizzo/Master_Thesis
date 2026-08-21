import numpy as np
import torch
import torch.nn as nn
from torch.func import functional_call, grad, vmap
import os
import re
import pandas as pd
import pytorch_lightning as pl
from tqdm import tqdm

from models.model_densenet import DenseNetModel

pl.seed_everything(42)

def calculate_gradnorm(models, train_dataloader, device, num_classes):
    scores = {"id": [], "epoch": [], "label": []}

    for c in range(num_classes):
        scores[f"c{c}_grad_norm"] = []

    CE = nn.CrossEntropyLoss(reduction="none").to(device)

    for X, label, id in tqdm(train_dataloader, desc="Processing Samples", unit="sample"):
        X, label = X.to(device), label.to(device)

        for epoch_idx, net in enumerate(models):
            net.eval()

            for c in range(num_classes):
                net.zero_grad()

                scores = metrics_for_batch(
                    net=net,
                    X=X,
                    labels=label,
                    ids=id,
                    epoch=epoch_idx,
                    c=c,
                    scores=scores,
                    loss_fct=CE,
                    device=device,
                )

    return pd.DataFrame(scores)

def metrics_for_batch(net, X, labels, ids, epoch, c, scores, loss_fct, device):
    params = {k: v.detach() for k, v in net.named_parameters()}
    buffers = {k: v.detach() for k, v in net.named_buffers()}

    def compute_loss(params, buffers, sample, target):
        sample = sample.unsqueeze(0)
        target = target.unsqueeze(0)
        logits = functional_call(net, (params, buffers), (sample,))
        loss = loss_fct(logits, target)
        return loss[0]

    ft_compute_sample_grad = vmap(grad(compute_loss), in_dims=(None, None, 0, 0))
    c_tensor = torch.full_like(labels, c).to(device)
    grads = ft_compute_sample_grad(params, buffers, X, c_tensor)
    del ft_compute_sample_grad

    for i in range(len(X)):
        if c == 0:
            scores["id"].append(str(ids[i]))
            scores["epoch"].append(epoch)
            scores["label"].append(int(labels[i].detach().cpu().item()))

        grad_i = [grads[name][i].flatten() for name in grads]
        grad_norm = torch.linalg.norm(torch.cat(grad_i))
        scores[f"c{c}_grad_norm"].append(float(grad_norm.detach().cpu().item()))

    return scores

def calculate_glare(grad_norms):
    n_classes = grad_norms['label'].max() + 1
    n_epoch = grad_norms["epoch"].max() + 1
    
    print(f"DEBUG: n_classes={n_classes}, n_epoch={n_epoch}")
    
    scores_list = []
    
    def max_index_excluding_class_x(lst, x):
        filtered_list = np.delete(lst, x)
        max_value = max(filtered_list)
        max_index = np.where(lst==max_value)[0][0]
        return max_index

    # Calculate GLARE scores for each sample ID
    for sample_id in grad_norms['id'].unique():
        sample_data = grad_norms[grad_norms['id'] == sample_id]
        
        if len(sample_data) == 0:
            continue
            
        label = int(sample_data['label'].iloc[0])
        
        glare = np.zeros(n_classes)
        grad_norm_in_min_epochs = np.zeros(n_classes)
        grad_norm_total = np.zeros(n_classes)
        
        # Process each epoch for sample
        for epoch in range(n_epoch):
            epoch_data = sample_data[sample_data['epoch'] == epoch]
            
            if len(epoch_data) == 0:
                continue
                
            # Get gradient norms for all classes in this epoch
            epoch_values = np.zeros(n_classes)
            for c in range(n_classes):
                col_name = f"c{c}_grad_norm"
                if col_name in epoch_data.columns:
                    epoch_values[c] = epoch_data[col_name].iloc[0]
            
            # Find class with minimum gradient
            min_class = np.argmin(epoch_values)
            glare[min_class] += 1
            grad_norm_in_min_epochs[min_class] += epoch_values[min_class]
            grad_norm_total += epoch_values
        
        # Skip if no valid data
        if np.sum(grad_norm_total) == 0:
            continue
            
        alt_class_glare = max_index_excluding_class_x(glare, label)
        
        # Calculate GLAREX scores
        glarex = np.zeros(n_classes)
        for c in range(n_classes):
            if glare[c] > 0:
                avg = grad_norm_in_min_epochs[c] / glare[c]
                inv_avg = 1.0 / avg if avg > 0 else 0
                glarex[c] = glare[c] + 0.01 * inv_avg
            else:
                avg = grad_norm_total[c] / n_epoch
                inv_avg = 1.0 / avg if avg > 0 else 0
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

    scores_df = pd.DataFrame(scores_list, columns=[
        "id", 
        "label", 
        "glare score", 
        "glarex score",
        "alternative class (glare)",
        "alternative glare score",
        "alternative class (glarex)",
        "alternative glarex score"])
    
    return scores_df

def compute_glare(dataloader, num_classes, weights_dir, spec):
    pattern = re.compile(r'epoch_\d+$')  

    weights = [
        os.path.join(weights_dir, file)
        for file in os.listdir(weights_dir)
        if pattern.match(file)
    ]
    weights.sort(key=lambda x: int(x.rsplit('_', 1)[-1]))

    # Force use of only GPU 0 (Quadro RTX 5000) 
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # Single GPU Info
    print(f"Verwende GPU 0 für stabile GLARE-Berechnung: {torch.cuda.get_device_name(0)}")
    print(f"Grund: Optimale Performance ohne DataParallel-Komplexität")

    # Load models for each epoch
    models = []
    for index, weight_path in enumerate(weights):
        full_model = DenseNetModel(
            num_classes=num_classes,
            spec=spec,
            weights_dir=weights_dir
        )
        state_dict = torch.load(weight_path, map_location=device)
        full_model.load_state_dict(state_dict)
        net = full_model.model.to(device)
        
        # No DataParallel 
        print(f"  Epoche {index}: Model auf GPU 0 geladen")
        
        models.append(net)

    grad_norms = calculate_gradnorm(models=models, train_dataloader=dataloader, device=device, num_classes=num_classes)
    scores = calculate_glare(grad_norms)
    
    return scores, grad_norms