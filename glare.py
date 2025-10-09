
import numpy as np
import torch
import torch.nn as nn
from torch.func import functional_call, grad, vmap
import os
import re
import pandas as pd
import pytorch_lightning as pl
from tqdm import tqdm

from models.model import Model


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
    
    for X, label, id in tqdm(train_dataloader, desc="Processing Samples", unit="sample"): # before: X, label, id, _
        X, label = X.to(device), label.to(device) 

        for index, net in enumerate(models):
            net.eval()
            
            for c in range(num_classes):
                net.zero_grad()  
                
                scores = metrics_for_batch(net=net, X=X, labels=label, ids=id, epoch=index, c=c, scores=scores, loss_fct=CE, device=device)

    scores = pd.DataFrame(scores)

    return scores



def metrics_for_batch(net, X, labels, ids, epoch, c, scores, loss_fct, device):
    params = {k: v.detach() for k, v in net.named_parameters()}
    buffers = {k: v.detach() for k, v in net.named_buffers()}
    
    def compute_loss(params, buffers, sample, target):
        sample = sample.unsqueeze(0)
        target = target.unsqueeze(0)

        logits = functional_call(net, (params, buffers), (sample,))
        loss = loss_fct(logits, target)
        return loss[0]

    ft_compute_grad = grad(compute_loss)
    ft_compute_sample_grad = vmap(ft_compute_grad, in_dims=(None, None, 0, 0))#, randomness="same")
    c_tensor = torch.full_like(labels, c).to(device)
    grads = ft_compute_sample_grad(params, buffers, X, c_tensor)
    del ft_compute_sample_grad

    for i in range(len(X)):

        if c == 0:
            scores["id"].append(ids[i]) # neu
            #scores["id"].append(int(ids[i].detach().cpu().item()))
            scores["epoch"].append(epoch)
            scores["label"].append(int(labels[i].detach().cpu().item()))

        grad_i = [grads[name][i].flatten() for name in grads]
        flattened_grad = torch.cat(grad_i)

        grad_norm = torch.linalg.norm(flattened_grad.flatten())

        scores[f"c{c}_grad_norm"].append(float(grad_norm.detach().cpu().item()))

    return scores


def calculate_glare(grad_norms):
    n_classes = grad_norms['label'].max() + 1
    class_indices = range(n_classes)
    gradient_columns = [f"c{c}_grad_norm" for c in class_indices]

    class_matrices = {}

    for col in gradient_columns:
        matrix = grad_norms.pivot(index="id", columns="epoch", values=col)

        matrix = matrix.reset_index()
        matrix['class'] = col
        matrix = matrix.merge(grad_norms[['id', 'label']].drop_duplicates(), on='id', how='left')
        class_matrices[col] = matrix


    grad_norms_per_class = pd.concat(class_matrices.values(), ignore_index=True)


    n_epoch = grad_norms["epoch"].max() + 1
    scores_list = []

    def max_index_excluding_class_x(lst, x):
        filtered_list = np.delete(lst, x)
        max_value = max(filtered_list)
        max_index = np.where(lst==max_value)[0][0]
        return max_index


    for _, group_df in grad_norms_per_class.groupby("id"):

        glare = np.zeros(n_classes)
        min_class_sequence = []
        label = int(group_df["label"].values[0])
        grad_norm_in_min_epochs = np.zeros(n_classes)
        grad_norm_total = np.zeros(n_classes)

        # Get the lowest grad norm class for each epoch
        for epoch in range(n_epoch):
            values = group_df[epoch]
            min_class = np.argmin(values)
            glare[min_class] += 1
            grad_norm_in_min_epochs[min_class] += values.values[min_class]
            min_class_sequence.append(min_class)
            grad_norm_total += values.values
        
        alt_class_glare = max_index_excluding_class_x(glare, label)
        
        # Compute avg lowest value for each class (only where it was the minimum)
        glarex = np.zeros(n_classes)

        for c in range(n_classes):
            if glare[c] > 0:
                avg = grad_norm_in_min_epochs[c] / glare[c]
                inv_avg = 1.0 / (avg)
                glarex[c] = glare[c] + 0.01 * inv_avg
            else:
                avg = grad_norm_total[c] / n_epoch
                inv_avg = 1.0 / (avg)
                glarex[c] = glare[c] + 0.01 * inv_avg

        alt_class_glarex = max_index_excluding_class_x(glarex, label)
        
        scores_list.append([
            group_df["id"].values[0], 
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

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create models for each epoch
    models = [] 
    for index, weight_path in enumerate(weights):
        net = Model(
            num_classes=num_classes,
            spec=spec,
            weights_dir=weights_dir
        ).model.to(device)
        state_dict = torch.load(weight_path, map_location=device)
        new_state_dict = {k.replace('model.', ''): v for k, v in state_dict.items()}
        net.load_state_dict(new_state_dict, strict=False)

        models.append(net)


    grad_norms = calculate_gradnorm(models=models, train_dataloader=dataloader, device=device, num_classes=num_classes)
    scores = calculate_glare(grad_norms)

    
    return scores, grad_norms
