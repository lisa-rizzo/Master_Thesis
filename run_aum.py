"""AUM forward sweep: one forward pass per (sample x checkpoint), no gradients.

AUM (Area Under the Margin, Pleiss et al. 2020) per sample:
    AUM = mean_t ( z_y(w_t, x) - max_{j != y} z_j(w_t, x) )      (logit margin, stored label y)
Lower = more suspicious. The final score is computed in evaluation/evaluate_run.py (aum_scores);
this script stores the raw logits per sample and checkpoint.

Deviation from the paper: eval() on the unaugmented sample at stored checkpoints,
not the forward pass during training; no threshold samples (ranking metrics need no threshold).

The same sweep also yields (all computed in evaluate_run.py):
  * TracIn on the last layer (tracin_ll) — closed form via p_all and h_norm:
        ||dCE/dW||^2 + ||dCE/db||^2 = ||p - e_c||^2 * (||h||^2 + 1)   (helper/last_layer.py)
  * Confidence baselines conf_p (mean p_y) and conf_win (# checkpoints where pred == y)

Output: <out-dir>/confidence.pkl (one row per sample x checkpoint) + confidence_info.json.

    python run_aum.py --weights-dir <run>/weights --out-dir <out>/aum
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
import torch.nn.functional as F
from monai.transforms import Compose, NormalizeIntensity
from tqdm import tqdm

from data.dataloader_VerSe_scoring import VerSeDataLoader
from helper import paths
from helper.checkpoints import find_epoch_weights, load_epoch_model, select_checkpoints


class ImageTransform:
    def __init__(self, mean, std):
        self.transforms = Compose([
            NormalizeIntensity(subtrahend=mean, divisor=std, channel_wise=True),
        ])

    def __call__(self, data):
        return self.transforms(data)


def parse_args():
    p = argparse.ArgumentParser(description="AUM-Forward-Sweep (Logits je Sample x Checkpoint)")
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


@torch.no_grad()
def forward_checkpoint(net, loader, epoch_idx, device, rows):
    """
    One forward pass over all samples for exactly one checkpoint.

    Stores logits, the full softmax vector (p_all), and the norm of the penultimate
    feature h (h_norm, output of class_layers.flatten, i.e. before the last linear layer)
    — evaluate_run.py uses these to compute TracIn on the last layer.
    """
    feats = {}
    hook = net.class_layers.flatten.register_forward_hook(
        lambda _m, _inp, out: feats.__setitem__("h", out.detach()))
    try:
        for X, labels, ids in loader:
            X = X.to(device, non_blocking=True)
            y = labels.to(device, non_blocking=True)
            logits = net(X)
            h_norm = feats["h"].float().norm(dim=1)
            probs = F.softmax(logits, dim=1)
            loss = F.cross_entropy(logits, y, reduction="none")
            pred = logits.argmax(dim=1)
            p_own = probs.gather(1, y.view(-1, 1)).squeeze(1)
            p_max = probs.max(dim=1).values
            for i, sid in enumerate(ids):
                rows.append({
                    "id": sid,
                    "epoch": epoch_idx,
                    "label": int(y[i].item()),
                    "pred": int(pred[i].item()),
                    "correct": bool(pred[i].item() == y[i].item()),
                    "p_own": float(p_own[i].item()),
                    "p_max": float(p_max[i].item()),
                    "loss": float(loss[i].item()),
                    "p_all": probs[i].float().cpu().numpy().astype(np.float32),
                    # raw logits for AUM (z_y - max_{j!=y} z_j): deriving the margin from p_all
                    # log p_y - log p_j, saettigt aber in float32, sobald p_y == 1.0 wird
                    "logits": logits[i].float().cpu().numpy().astype(np.float32),
                    "h_norm": float(h_norm[i].item()),
                })
    finally:
        hook.remove()


def main():
    args = parse_args()

    weights_dir = Path(args.weights_dir).resolve()
    spec_path = Path(args.spec_path) if args.spec_path else weights_dir.parent / "spec.json"
    with open(spec_path, "r") as f:
        spec = json.load(f)

    # batch_size 1 as in GLARE: custom_collate pads each batch to its largest volume.
    # With batch>1 a sample's tensor depends on its neighbours, breaking comparability with GLARE.
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
    print(f"{len(epochs)} of {len(all_epochs)} checkpoints: {names}")

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

    start = time.time()
    rows = []
    # Checkpoint outer, samples inner: only one model on GPU at a time
    for i, w in enumerate(epochs):
        net = load_epoch_model(w, num_classes, spec, device)
        bar = tqdm(total=len(loader), desc=f"Ckpt {i + 1}/{len(epochs)} ({names[i]})", unit="sample")

        def counted(loader=loader, bar=bar):
            for b in loader:
                yield b
                bar.update(1)

        forward_checkpoint(net, counted(), i, device, rows)
        bar.close()
        del net
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

    df = pd.DataFrame(rows)
    out = out_dir / "confidence.pkl"
    df.to_pickle(out)

    elapsed = (time.time() - start) / 60
    info = {
        "n_samples": int(df.id.nunique()), "n_rows": len(df),
        "n_epochs": len(epochs), "checkpoints": names,
        "checkpoint_selection": args.checkpoints, "num_classes": num_classes,
        "label_mode": spec.get("label_mode", "region"), "split": args.split,
        "weights_dir": str(weights_dir), "spec_path": str(spec_path),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "host": os.uname().nodename, "elapsed_minutes": elapsed,
        "timestamp": datetime.now().isoformat(),
    }
    with open(out_dir / "confidence_info.json", "w") as f:
        json.dump(info, f, indent=2)

    print(f"\nDone: {df.id.nunique()} samples, {len(df)} rows, {elapsed:.1f} min")
    print(f"Geschrieben: {out}")


if __name__ == "__main__":
    main()
