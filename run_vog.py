"""VoG: Variance of Gradients (Agarwal, D'souza, Hooker, CVPR 2022) baseline.

For each sample x and checkpoint t, compute the input gradient of the pre-softmax logit
for the stored label y (the training label, not the prediction — as in the paper):

    S_t = d z_y(w_t, x) / d x                            (one volume per checkpoint)
    VoG = mean_voxel sqrt( mean_t (S_t - mean_t S)^2 )   (std over checkpoints, then mean)

Class normalisation (z-score per stored class, negated: lower = more suspicious) is applied
in evaluation/evaluate_run.py (vog_scores) over the full training set.

Unlike run_aum.py the loop is sample-outer, checkpoint-inner: VoG needs all K gradients
of ONE sample simultaneously (voxel-wise variance). K densenet169 models are 75 MB each,
so all checkpoints fit on the GPU (~2 GB for 8 checkpoints), and each file is read once.
Gradients w.r.t. input only: parameters get requires_grad=False, saving weight gradients.

Windows: 'vog' over all selected checkpoints, 'vog_early' / 'vog_late' over the first /
second half (the paper distinguishes early and late training). Also stored per checkpoint:
||S_t|| and full logits.

Output: <out-dir>/vog.pkl (one row per sample) + vog_info.json.

    python run_vog.py --weights-dir <run>/weights --out-dir <out>/vog
"""

import argparse
import gc
import json
import os
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from data.dataloader_VerSe_scoring import VerSeDataLoader
from helper import paths
from helper.checkpoints import find_epoch_weights, load_epoch_model, select_checkpoints
from run_aum import ImageTransform


def parse_args():
    p = argparse.ArgumentParser(description="Variance of Gradients (VoG)")
    p.add_argument("--weights-dir", required=True, help="Directory with epoch_0, epoch_1, ...")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--spec-path", default=None, help="Default: <weights-dir>/../spec.json")
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--checkpoints", default=None,
                   help="'all' (default), 'auto:<n>', or comma-separated list of basenames")
    p.add_argument("--limit-samples", type=int, default=None, help="only the first N samples (smoke-test)")
    p.add_argument("--label-mode", default=None)
    p.add_argument("--split", choices=["train", "val"], default="train")
    p.add_argument("--allow-cpu", action="store_true",
                   help="Continue without a GPU (implicitly allowed with --limit-samples)")
    p.add_argument("--label-overrides", default=None,
                   help="Override spec['label_overrides'] (CSV id,label; empty string = none)")
    return p.parse_args()


def vog_windows(n_ckpt: int) -> dict:
    """Name -> Checkpoint-Indizes. Halbierung erst ab 4 Checkpoints (je Haelfte >= 2)."""
    w = {"vog": list(range(n_ckpt))}
    if n_ckpt >= 4:
        h = n_ckpt // 2
        w["vog_early"] = list(range(h))
        w["vog_late"] = list(range(h, n_ckpt))
    return w


def input_gradients(nets, x, y):
    """-> G (K, *x.shape[1:]) = d z_y / d x je Checkpoint, logits (K, C)."""
    grads, logits = [], []
    for net in nets:
        xi = x.detach().clone().requires_grad_(True)
        out = net(xi)
        g, = torch.autograd.grad(out[0, y], xi)
        grads.append(g[0])
        logits.append(out[0].detach())
    return torch.stack(grads), torch.stack(logits)


def vog_from_grads(G, windows):
    """Voxel-wise std across checkpoints (population std as in the paper), averaged over voxels."""
    return {name: float(G[idx].std(dim=0, unbiased=False).mean().item())
            for name, idx in windows.items()}


def main():
    args = parse_args()

    weights_dir = Path(args.weights_dir).resolve()
    spec_path = Path(args.spec_path) if args.spec_path else weights_dir.parent / "spec.json"
    with open(spec_path, "r") as f:
        spec = json.load(f)

    # batch_size 1 as in GLARE: custom_collate pads to the largest volume in the batch;
    # zero-padding would otherwise enter the voxel mean.
    spec["batch_size"] = 1
    paths.portable_spec(spec)
    if args.label_mode is not None:
        spec["label_mode"] = args.label_mode
    if args.label_overrides is not None:
        spec["label_overrides"] = args.label_overrides or None

    spec_data = {"mean": [-111.3885], "std": [406.3665]}
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_epochs = find_epoch_weights(str(weights_dir))
    epochs = select_checkpoints(all_epochs, args.checkpoints)
    names = [os.path.basename(e) for e in epochs]
    windows = vog_windows(len(epochs))
    print(f"{len(epochs)} of {len(all_epochs)} checkpoints | windows: "
          + ", ".join(f"{k}={names[v[0]]}..{names[v[-1]]}" for k, v in windows.items()))
    if len(epochs) < 2:
        raise SystemExit("VoG braucht mindestens 2 Checkpoints")

    transform = ImageTransform(mean=spec_data["mean"], std=spec_data["std"])
    data_module = VerSeDataLoader(spec=spec, train_transforms=transform,
                                  val_transforms=transform, num_workers=args.num_workers)
    data_module.setup()
    if args.limit_samples is not None:
        ds = data_module.train_dataset if args.split == "train" else data_module.val_dataset
        ds.file_paths = ds.file_paths[:args.limit_samples]
        print(f"SMOKE-TEST: limited to {len(ds.file_paths)} samples")

    if not torch.cuda.is_available() and not (args.allow_cpu or args.limit_samples):
        raise SystemExit("No CUDA GPU visible. Aborting instead of silent CPU fallback "
                         "(--allow-cpu forces CPU; implicitly allowed with --limit-samples).")

    # shuffle=False, fixed order, labels/ids identical to training
    loader = data_module.scoring_dataloader(split=args.split)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    num_classes = data_module.num_classes

    nets = []
    for w in epochs:
        net = load_epoch_model(w, num_classes, spec, device)     # eval(): BN running stats, no dropout
        for p in net.parameters():
            p.requires_grad_(False)
        nets.append(net)

    start = time.time()
    rows = []
    for X, labels, ids in tqdm(loader, desc="VoG", unit="sample"):
        if X.shape[0] != 1:
            raise RuntimeError("run_vog.py requires batch_size 1")
        X = X.to(device, non_blocking=True)
        y = int(labels[0].item())
        G, logits = input_gradients(nets, X, y)
        rows.append({
            "id": str(ids[0]),
            "label": y,
            "n_vox": int(G[0].numel()),
            **vog_from_grads(G, windows),
            "grad_l2": G.flatten(1).norm(dim=1).float().cpu().numpy().astype(np.float32),
            "logits": logits.float().cpu().numpy().astype(np.float32),
        })
        del G

    df = pd.DataFrame(rows)
    out = out_dir / "vog.pkl"
    df.to_pickle(out)

    elapsed = (time.time() - start) / 60
    info = {
        "n_samples": int(df.id.nunique()) if len(df) else 0, "n_epochs": len(epochs),
        "checkpoints": names, "checkpoint_selection": args.checkpoints,
        "windows": {k: [names[i] for i in v] for k, v in windows.items()},
        "num_classes": num_classes, "label_mode": spec.get("label_mode", "region"),
        "split": args.split, "weights_dir": str(weights_dir), "spec_path": str(spec_path),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "host": os.uname().nodename, "elapsed_minutes": elapsed,
        "samples_per_second": len(df) / (elapsed * 60) if elapsed else 0.0,
        "max_gpu_mem_gb": torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else None,
        "timestamp": datetime.now().isoformat(),
    }
    with open(out_dir / "vog_info.json", "w") as f:
        json.dump(info, f, indent=2)

    del nets
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()
    print(f"\nDone: {len(df)} samples, {elapsed:.1f} min")
    print(f"Geschrieben: {out}")


if __name__ == "__main__":
    main()
