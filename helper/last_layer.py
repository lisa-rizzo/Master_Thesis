"""
Last-layer gradient norms in closed form (TracIn-LL).

For softmax-CE over a linear head z = W h + b, for target class c:

    dCE/dz = p - e_c
    dCE/dW = (p - e_c) h^T      ->  ||dCE/dW||_F = ||p - e_c|| * ||h||
    dCE/db = p - e_c

Therefore  ||grad_{W,b} CE||  =  ||p - e_c|| * sqrt(||h||^2 + 1).

Two consequences relevant for comparing GLARE against plain confidence:
  1. ||p - e_c||^2 = sum_j p_j^2 - 2 p_c + 1  ->  argmin_c ||p - e_c|| = argmax_c p_c.
     The win-count on the classifier head is exactly the model's prediction.
  2. The factor sqrt(||h||^2 + 1) is the same for all candidate classes of a sample;
     it does not change argmin but does affect the absolute value (GLAREx tiebreaker, margin).
What GLARE sees beyond this with the full parameter gradient comes solely from
back-propagating through the network's Jacobian.

Input is the output of run_aum.py (columns p_all, h_norm). The output has the same
schema as GLARE grad_norms in region mode (id, epoch, label, c{c}_grad_norm),
so calculate_glare / evaluate_run.py can consume it unchanged.
"""
import numpy as np
import pandas as pd


def last_layer_grad_norms(conf: pd.DataFrame) -> pd.DataFrame:
    """confidence DataFrame -> DataFrame in grad_norms schema (id, epoch, label, c*_grad_norm)."""
    if "p_all" not in conf.columns or "h_norm" not in conf.columns:
        raise ValueError("confidence data missing p_all/h_norm columns -- "
                         "recompute with the current run_aum.py.")
    P = np.stack(conf["p_all"].to_numpy()).astype(np.float64)       # (n, K)
    scale = np.sqrt(conf["h_norm"].to_numpy(dtype=np.float64) ** 2 + 1.0)
    sq = (P ** 2).sum(axis=1, keepdims=True)                          # sum_j p_j^2
    dist = np.sqrt(np.clip(sq - 2.0 * P + 1.0, 0.0, None))            # ||p - e_c|| je c
    out = conf[["id", "epoch", "label"]].copy()
    for c in range(P.shape[1]):
        out[f"c{c}_grad_norm"] = dist[:, c] * scale
    return out


def load_confidence_dir(conf_dir) -> pd.DataFrame:
    """Load confidence.pkl (run_aum.py) or all confidence_shard*.pkl files from a run."""
    from pathlib import Path
    files = sorted(Path(conf_dir).glob("confidence*.pkl"))
    if not files:
        raise FileNotFoundError(f"no confidence*.pkl found in {conf_dir}")
    return pd.concat([pd.read_pickle(f) for f in files], ignore_index=True)
