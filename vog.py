"""
VOG (Variance of Gradients) mislabel detection — post-hoc, feature-level.

Method: Khanal et al. (MICCAI 2024), feature-level variant.
  For each training sample and epoch checkpoint, compute the gradient of the
  cross-entropy loss w.r.t. the flattened feature vector (after global average
  pooling, before the classifier head).  Variance of this gradient across epochs
  is the suspicion signal — high variance = model inconsistent about this sample
  = likely mislabeled.

Score direction: HIGHER vog_score = more suspicious (opposite of GLARE).
"""

import os
import re

import pandas as pd
import torch
import torch.nn.functional as F
from tqdm import tqdm

from models.model_densenet import DenseNetModel


def get_feature_grads_for_epoch(net, dataloader, device):
    """
    For one checkpoint, compute dCE/d(flat_feature) for every training sample.

    The gradient is taken w.r.t. the flattened 1-D feature vector that sits
    between global-average-pooling and the linear classifier head.  Running the
    backbone under no_grad and detaching its output keeps the computation graph
    small (no weight gradients accumulate).

    Returns
    -------
    grads  : dict  sample_id -> 1-D CPU tensor  [D]
    labels : dict  sample_id -> int label
    """
    grads = {}
    labels_out = {}
    net.eval()

    for X, labels, ids in tqdm(dataloader, desc="  samples", leave=False):
        X = X.to(device)

        for i in range(len(X)):
            x_i = X[i : i + 1]
            label_i = int(labels[i].item())
            sample_id = ids[i]

            # Backbone: no grad needed, detach to isolate the computation graph.
            with torch.no_grad():
                feat_map = net.features(x_i)

            # Classifier head: grad flows from loss back to feat_leaf.
            feat_leaf = feat_map.detach().requires_grad_(True)
            relu_out = F.relu(feat_leaf)  # non-inplace: class_layers.relu has inplace=True
            pool_out = net.class_layers.pool(relu_out)
            flat_out = net.class_layers.flatten(pool_out)  # [1, D]
            flat_out.retain_grad()
            logits = net.class_layers.out(flat_out)  # [1, num_classes]

            loss = F.cross_entropy(
                logits, torch.tensor([label_i], device=device)
            )
            loss.backward()

            grads[sample_id] = flat_out.grad[0].detach().cpu()
            labels_out[sample_id] = label_i

    return grads, labels_out


def calculate_vog(grads_per_sample, labels_per_sample):
    """
    Compute a scalar VOG score per sample from gradient vectors collected
    across E epoch checkpoints.

    VOG_i = (1/D) * sum_d  sqrt( (1/E) * sum_e (S_ie - mu_i_d)^2 )
          = mean_d( std_e(S_ie_d) )      [element-wise std, averaged over dims]

    Parameters
    ----------
    grads_per_sample  : dict  sample_id -> list of [D] tensors (length E)
    labels_per_sample : dict  sample_id -> int

    Returns
    -------
    DataFrame with columns ["id", "label", "vog_score"]
    """
    rows = []
    for sample_id, grad_list in grads_per_sample.items():
        grads_stack = torch.stack(grad_list)          # [E, D]
        mu = grads_stack.mean(dim=0)                  # [D]
        variance = ((grads_stack - mu) ** 2).mean(dim=0)  # [D]
        vog_score = variance.sqrt().mean().item()     # scalar

        rows.append(
            {
                "id": sample_id,
                "label": labels_per_sample[sample_id],
                "vog_score": vog_score,
            }
        )

    return pd.DataFrame(rows)


def compute_vog(dataloader, weights_dir, spec, num_classes=4, t=5):
    """
    Post-hoc VOG computation on saved epoch checkpoints.

    Parameters
    ----------
    dataloader  : training dataloader (batch_size should be 1)
    weights_dir : directory containing epoch_0, epoch_1, … checkpoints
    spec        : model spec dict (passed to DenseNetModel)
    num_classes : number of output classes
    t           : number of most-recent epoch checkpoints to use (default 5)

    Returns
    -------
    DataFrame with columns ["id", "label", "vog_score"]
    """
    epoch_pattern = re.compile(r"^epoch_(\d+)$")
    all_weights = sorted(
        [
            os.path.join(weights_dir, f)
            for f in os.listdir(weights_dir)
            if epoch_pattern.match(f)
        ],
        key=lambda x: int(x.rsplit("_", 1)[-1]),
    )

    if len(all_weights) == 0:
        raise RuntimeError(f"No epoch checkpoints found in {weights_dir}")

    # Use the last t checkpoints (sliding-window end state).
    epoch_weights = all_weights[-t:]
    print(
        f"Using {len(epoch_weights)} of {len(all_weights)} checkpoints: "
        f"{[os.path.basename(w) for w in epoch_weights]}"
    )

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Device: {torch.cuda.get_device_name(0) if device.type == 'cuda' else 'cpu'}")

    all_grads = {}   # sample_id -> list[Tensor]
    all_labels = {}  # sample_id -> int

    for epoch_idx, weight_path in enumerate(epoch_weights):
        print(
            f"\nEpoch checkpoint {epoch_idx + 1}/{len(epoch_weights)}: "
            f"{os.path.basename(weight_path)}"
        )

        full_model = DenseNetModel(
            num_classes=num_classes, spec=spec, weights_dir=weights_dir
        )
        state_dict = torch.load(weight_path, map_location=device)
        full_model.load_state_dict(state_dict)
        net = full_model.model.to(device)

        epoch_grads, epoch_labels = get_feature_grads_for_epoch(
            net, dataloader, device
        )

        for sid, grad in epoch_grads.items():
            if sid not in all_grads:
                all_grads[sid] = []
                all_labels[sid] = epoch_labels[sid]
            all_grads[sid].append(grad)

        del full_model, net
        if device.type == "cuda":
            torch.cuda.empty_cache()

    print(f"\nComputing VOG scores for {len(all_grads)} samples...")
    return calculate_vog(all_grads, all_labels)
