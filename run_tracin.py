"""
TracIn self-influence scoring for mislabel detection.

TracInCP self-influence (Pruthi et al., NeurIPS 2020):
    S_self(z) = sum_{t=0}^{T-1}  eta_t * ||grad_theta L(z; theta_t)||^2

High score → the sample strongly "taught" the model its own (potentially wrong) label
→ likely mislabelled.  This is the analogue of the GLARE score but from a different
theoretical angle; both should be compared on the same evaluation set.

Usage:
    python run_tracin.py
"""

import gc
import json
import os
import re
import time
from datetime import datetime
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
from monai.transforms import Compose, NormalizeIntensity
from torch.func import functional_call, grad, vmap
from tqdm import tqdm

from dataloaders.dataloader_VerSe_new import VerSeDataLoader
from models.model_densenet import DenseNetModel


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

SPEC_PATH   = "/home/student/lisa_ma/results/VerSe_classifier/label_override_8ep_23092026_2135/spec.json"
WEIGHTS_DIR = "/home/student/lisa_ma/results/VerSe_classifier/label_override_8ep_23092026_2135/weights"
RESULTS_BASE_DIR = os.path.join("results", "tracin")

MEAN = [-111.3885]
STD  = [406.3665]

NUM_CLASSES = 4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class ImageTransform:
    def __init__(self, mean, std):
        self.transforms = Compose([NormalizeIntensity(subtrahend=mean, divisor=std, channel_wise=True)])
    def __call__(self, data):
        return self.transforms(data)


def load_model(weights_path: str, spec: dict) -> nn.Module:
    """Load inner MONAI DenseNet from a plain state-dict checkpoint."""
    wrapper = DenseNetModel(num_classes=NUM_CLASSES, spec=spec, weights_dir=WEIGHTS_DIR)
    state = torch.load(weights_path, map_location=DEVICE)
    wrapper.load_state_dict(state)
    net = wrapper.model.to(DEVICE)
    net.eval()
    return net


def sq_grad_norm_batch(net: nn.Module, X: torch.Tensor, labels: torch.Tensor) -> list[float]:
    """
    Return per-sample squared gradient norms using the true labels.
    Uses the same vmap(grad(...)) pattern as glare.py.
    """
    CE = nn.CrossEntropyLoss(reduction="none").to(DEVICE)
    params  = {k: v.detach() for k, v in net.named_parameters()}
    buffers = {k: v.detach() for k, v in net.named_buffers()}

    def compute_loss(params, buffers, sample, target):
        sample = sample.unsqueeze(0)
        target = target.unsqueeze(0)
        logits = functional_call(net, (params, buffers), (sample,))
        return CE(logits, target)[0]

    ft_grad = vmap(grad(compute_loss), in_dims=(None, None, 0, 0))
    grads = ft_grad(params, buffers, X, labels)

    sq_norms = []
    for i in range(len(X)):
        flat = torch.cat([grads[name][i].flatten() for name in grads])
        sq_norms.append(float((flat * flat).sum().detach().cpu()))
    return sq_norms


# ---------------------------------------------------------------------------
# Main TracIn loop
# ---------------------------------------------------------------------------

def compute_tracin(train_loader, weights_dir: str, spec: dict) -> pd.DataFrame:
    """
    Compute TracIn self-influence for every training sample.

    Outer loop: checkpoints (loaded once each).
    Inner loop: batches.
    """
    # Discover epoch checkpoints
    epoch_re = re.compile(r'^epoch_(\d+)$')
    epoch_numbers = sorted(
        int(m.group(1))
        for f in os.listdir(weights_dir)
        if (m := epoch_re.match(f))
    )

    # Load learning-rate schedule saved by the trainer.
    # lr.json[i] is logged in on_train_epoch_end BEFORE Lightning steps the
    # scheduler, so lr.json[i] = LR after epoch i's step = LR used during epoch i+1.
    # Verified empirically: LR before any step = spec["lr"] = 1e-4;
    # lr.json[0] = 8.7625e-05 = first post-step value.
    # Therefore: epoch_0 used initial LR; epoch_i (i>0) used lr.json[i-1]; lr.json[-1] unused.
    lr_path = os.path.join(weights_dir, "lr.json")
    with open(lr_path) as f:
        lr_schedule = json.load(f)

    initial_lr = spec.get("lr", 1e-4)
    # epoch_idx → LR used when training that epoch (and producing that checkpoint)
    epoch_to_lr: dict[int, float] = {0: initial_lr}
    for i, lr_val in enumerate(lr_schedule[:-1]):  # drop last entry: no epoch T+1
        epoch_to_lr[i + 1] = lr_val

    print(f"Checkpoints: {epoch_numbers}")
    print(f"Epoch → LR mapping: {epoch_to_lr}")

    # Accumulators keyed by sample_id
    accum: dict[str, dict] = {}   # sample_id → {"label": int, "score": float, "ep_{i}": float}

    for epoch_idx in epoch_numbers:
        weight_path = os.path.join(weights_dir, f"epoch_{epoch_idx}")
        eta = epoch_to_lr[epoch_idx]
        print(f"\n--- Epoch {epoch_idx}  (η={eta:.3e}) ---")

        net = load_model(weight_path, spec)

        for X, labels, ids in tqdm(train_loader, desc=f"ep{epoch_idx}", unit="batch"):
            X      = X.to(DEVICE)
            labels = labels.to(DEVICE)

            sq_norms = sq_grad_norm_batch(net, X, labels)

            for i, sid in enumerate(ids):
                if sid not in accum:
                    accum[sid] = {"label": int(labels[i].cpu()), "score": 0.0}
                accum[sid]["score"]           += eta * sq_norms[i]
                accum[sid][f"ep{epoch_idx}_sq_grad_norm"] = sq_norms[i]

        # Free GPU memory between checkpoints
        del net
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

    rows = [{"sample_id": sid, **v} for sid, v in accum.items()]
    df = pd.DataFrame(rows)
    col_order = ["sample_id", "label", "score"] + sorted(
        c for c in df.columns if c.startswith("ep")
    )
    return df[col_order]


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    with open(SPEC_PATH) as f:
        spec = json.load(f)

    spec["dataset_dir"] = "/home/student/lisa_ma/prepared"
    spec["batch_size"]  = 1    # match GLARE run for fair comparison

    os.makedirs(RESULTS_BASE_DIR, exist_ok=True)
    now    = datetime.now()
    run_ts = now.strftime("%Y%m%d_%H%M%S")

    # Dataloader
    data_module = VerSeDataLoader(
        spec=spec,
        train_transforms=ImageTransform(MEAN, STD),
        val_transforms=ImageTransform(MEAN, STD),
        num_workers=6,
    )
    data_module.setup()
    train_loader = data_module.train_dataloader()

    n_samples = len(train_loader.dataset)
    print(f"Device: {DEVICE}")
    print(f"Training samples: {n_samples}")

    t0 = time.time()
    scores = compute_tracin(train_loader, WEIGHTS_DIR, spec)
    elapsed = time.time() - t0

    print(f"\nDone in {elapsed/60:.1f} min  ({n_samples/elapsed:.2f} samples/s)")
    print(scores[["sample_id", "label", "score"]].describe())

    # Save
    run_dir = os.path.join(RESULTS_BASE_DIR, run_ts)
    os.makedirs(run_dir, exist_ok=True)

    csv_path = os.path.join(run_dir, "tracin_scores.csv")
    pkl_path = os.path.join(run_dir, "tracin_scores.pkl")
    scores.to_csv(csv_path, index=False)
    scores.to_pickle(pkl_path)

    # Save run metadata
    meta = {
        "run_ts": run_ts,
        "spec_path": SPEC_PATH,
        "weights_dir": WEIGHTS_DIR,
        "n_samples": n_samples,
        "elapsed_seconds": elapsed,
        "device": str(DEVICE),
    }
    with open(os.path.join(run_dir, "run_info.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\nResults saved to {run_dir}/")
    print(f"  tracin_scores.csv   — {len(scores)} rows, columns: {list(scores.columns)}")
    print(f"  tracin_scores.pkl")
    print(f"  run_info.json")
    print("\nNote: higher 'score' = stronger self-influence = more likely mislabelled.")
    print("      (Opposite sign convention from GLARE, where lower score = mislabelled.)")
