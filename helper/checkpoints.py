"""
Checkpoint helpers for the baselines (AUM, VoG, TracIn).

Loads epoch checkpoints in the same order as GLARE, without importing the GLARE module.
"""

import os
import re

import numpy as np
import torch

from models.model_densenet import DenseNetModel


_EPOCH_RE = re.compile(r'^epoch_(\d+)$')
_EPOCH_STEP_RE = re.compile(r'^epoch_(\d+)_step_(\d+)$')


def checkpoint_sort_key(filename):
    """
    Sort key (epoch, step) for a checkpoint filename.

    `epoch_<n>` is written at the *end* of epoch n, so it sorts after all
    sub-epoch checkpoints of the same epoch — hence inf as the step.
    Returns None if the name is not a checkpoint.
    """
    m = _EPOCH_STEP_RE.match(filename)
    if m:
        return (int(m.group(1)), int(m.group(2)))
    m = _EPOCH_RE.match(filename)
    if m:
        return (int(m.group(1)), float('inf'))
    return None


def find_epoch_weights(weights_dir):
    """
    All checkpoints in weights_dir, in training order.

    Recognises `epoch_<n>` (end-of-epoch) and `epoch_<n>_step_<s>` (sub-epoch checkpoints).
    Sub-epoch checkpoints exist because the GLARE signal is concentrated in epoch 0
    (~1000 samples contradict their own label there; only 12 in epoch 19).
    Older runs without sub-epoch checkpoints are read unchanged.
    """
    entries = []
    for file in os.listdir(weights_dir):
        key = checkpoint_sort_key(file)
        if key is not None:
            entries.append((key, os.path.join(weights_dir, file)))
    entries.sort(key=lambda t: t[0])
    return [path for _, path in entries]


def select_checkpoints(paths, selection=None):
    """
    Select the checkpoints to actually score from the discovered list.

    selection:
      None / "all"      -> all (default, unchanged behaviour)
      "auto:<n>"        -> all sub-epoch checkpoints plus log-spaced epoch-end checkpoints,
                           at most n in total
      "a,b,c"           -> exactly these basenames, in training order

    Background for auto: GLARE cost is linear in checkpoint count, but late checkpoints
    contribute little. Measured over flagged samples, the stored label wins in epoch 0
    in 21.4% of cases and in epoch 19 in 99.0% — the tail samples only a converged model.
    Sub-epoch checkpoints from epoch 0 are therefore always kept; thinning applies to
    the epoch-end checkpoints.
    """
    if not selection or selection == "all":
        return list(paths)

    by_name = {os.path.basename(p): p for p in paths}

    if not selection.startswith("auto:"):
        wanted = [x.strip() for x in selection.split(",") if x.strip()]
        missing = [w for w in wanted if w not in by_name]
        if missing:
            raise SystemExit(f"Checkpoints not found: {missing}\nAvailable: {sorted(by_name)}")
        # return in training order, not input order
        return [p for p in paths if os.path.basename(p) in set(wanted)]

    n = int(selection.split(":", 1)[1])
    if n <= 0:
        raise SystemExit("auto:<n> requires n > 0")
    sub = [p for p in paths if "_step_" in os.path.basename(p)]
    ends = [p for p in paths if "_step_" not in os.path.basename(p)]

    if len(sub) >= n:
        keep = set(sub[:n])
    else:
        k = min(n - len(sub), len(ends))
        if k <= 0:
            keep = set(sub)
        elif k == len(ends):
            keep = set(sub) | set(ends)
        else:
            # log-spaced: dense early, sparse late — epoch 0 and last always included
            idx = sorted({int(round(x)) for x in np.geomspace(1, len(ends), k)})
            idx = [i - 1 for i in idx]
            for i in range(len(ends)):          # fill up if geomspace produced duplicates
                if len(idx) >= k:
                    break
                if i not in idx:
                    idx.append(i)
            idx = sorted(set(idx))[:k]
            if len(ends) - 1 not in idx:        # always keep the last checkpoint
                idx = idx[:-1] + [len(ends) - 1]
            keep = set(sub) | {ends[i] for i in idx}

    return [p for p in paths if p in keep]


def load_epoch_model(weight_path, num_classes, spec, device):
    """Load a single epoch checkpoint and return the bare nn.Module."""
    full_model = DenseNetModel(num_classes=num_classes, spec=spec, weights_dir=os.path.dirname(weight_path))
    state_dict = torch.load(weight_path, map_location=device)
    full_model.load_state_dict(state_dict)
    net = full_model.model.to(device)
    net.eval()
    return net
