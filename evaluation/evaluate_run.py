"""
Evaluate one GLARE run against the region-crossing ground truth (gt_mislabels_complete.xlsx).

One script for every run and label space (region / region5 / vertebra), so the numbers in the
thesis chapters come from the same code path:

  - recall / precision at fraction of training data checked (0.1 ... 20 %)
  - AUROC and AUPRC (average precision) per scorer
  - integer threshold table for the GLARE win-count (as in results.pdf Table 6.1)
  - per anomaly type (T12_missing, T13_extra, label_shift_*) and per dataset (dsname)
  - truncation to the first N checkpoints (results.pdf Section 6.3) for every scorer
  - optional: confidence baseline and closed-form last-layer GLARE from run_aum.py
    output (--aum-dir), and a simulated ground truth for the synthetic-noise runs
    (--sim-gt, E2)

Scorers (lower = more suspicious for all of them):
  glare      number of checkpoints in which the stored label has the minimum gradient norm
  glarex     glare + 0.01 / mean gradient norm
  margin     mean over checkpoints of (min_alt - own) / own
  conf_p     mean softmax probability of the stored label            (needs --aum-dir)
  conf_win   number of checkpoints in which the stored label is the argmax  (ditto)
  ll_glarex / ll_margin   the same GLAREx / margin on last-layer gradient norms  (ditto)

Published baselines (same orientation, lower = more suspicious):
  aum        Area Under the Margin (Pleiss et al. 2020): mean over checkpoints of
             z_y - max_{j!=y} z_j. Post hoc from the forward sweep run_aum.py (--aum-dir, column
             `logits`; older sweeps without it fall back to log p_all). Deviation from the paper:
             eval mode on the unaugmented sample at the checkpoints, not the in-training forward
             pass; no threshold samples (the ranking metrics need no threshold).
  tracin     -TracInCP self-influence (Pruthi et al. 2020): -sum_t eta_t ||grad l(w_t, z, y)||^2
             over all parameters. Free: GLARE's own-label column is exactly ||grad l||. eta_t =
             the LinearLR learning rate of the epoch the checkpoint closes (from spec.json).
             Training used Adam, not SGD -- the usual TracInCP approximation.
  tracin_ll  the same on the closed-form last-layer gradient norm        (needs --aum-dir)
  vog, vog_early, vog_late   -VoG (Agarwal et al. 2022), class-normalised (z-score per stored
             label) over the whole training set; all / first half / second half of the
             checkpoints. From run_vog.py (--vog-dir). Fixed windows, so not in the truncation table.

The integer glare score has large ties (90 % of samples at the maximum). Recall at a fixed
fraction is therefore reported as the *expected* recall under random tie-breaking inside the
bucket that the cut falls into, rather than depending on an arbitrary sort order.

Usage:
  .venv/bin/python evaluation/evaluate_run.py --run-dir <glare_run_dir>
      [--gt evaluation/gt_mislabels_complete.xlsx] [--aum-dir <dir>] [--vog-dir <dir>]
      [--sim-gt <xlsx>] [--out-dir <dir>] [--no-truncation]
"""
import argparse
import json
import re
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from helper import paths  # noqa: E402

FRACTIONS = (0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.10, 0.15, 0.20)
REGION_NAMES = ("cervical", "thoracic", "lumbar", "sacral")
REGION5_NAMES = ("cervical", "thoracic", "T13", "lumbar", "sacral")


# ----------------------------------------------------------------------------------------
# Loading
# ----------------------------------------------------------------------------------------
def newest(run_dir: Path, pattern: str) -> Path:
    files = sorted(run_dir.glob(pattern))
    if not files:
        raise FileNotFoundError(f"no {pattern} in {run_dir} (GLARE output with grad_norms_<ts>.pkl expected)")
    return files[-1]


def detect_space(grad_norms: pd.DataFrame, spec: dict | None) -> tuple[str, list[str], list[int] | None]:
    """-> (label_space, score columns, offsets or None)."""
    off = sorted([c for c in grad_norms.columns if re.fullmatch(r"off[+-]\d+_grad_norm", c)],
                 key=lambda c: int(c[3:].split("_")[0]))
    if off:
        return "vertebra", off, [int(c[3:].split("_")[0]) for c in off]
    cls = sorted([c for c in grad_norms.columns if re.fullmatch(r"c\d+_grad_norm", c)],
                 key=lambda c: int(c[1:].split("_")[0]))
    mode = (spec or {}).get("label_mode")
    if mode not in ("region", "region5"):          # no spec (e.g. notebooks): infer from the class count
        mode = "region5" if len(cls) == 5 else "region"
    return mode, cls, None


def to_tensor(df: pd.DataFrame, cols: list[str]):
    """Long (id x epoch) frame -> ids, labels, V[n_id, n_ckpt, K]. Requires a complete grid."""
    df = df.sort_values(["id", "epoch"], kind="mergesort")
    ids = df["id"].to_numpy()
    n_ckpt = int(df["epoch"].nunique())
    if len(df) % n_ckpt:
        raise ValueError("grad_norms is not a complete id x checkpoint grid")
    n_id = len(df) // n_ckpt
    V = df[cols].to_numpy(dtype=float).reshape(n_id, n_ckpt, len(cols))
    ids = ids.reshape(n_id, n_ckpt)[:, 0]
    labels = df["label"].to_numpy().reshape(n_id, n_ckpt)[:, 0].astype(int)
    return ids, labels, V


# ----------------------------------------------------------------------------------------
# Vectorised GLARE
# ----------------------------------------------------------------------------------------
def glare_scores(V: np.ndarray, own_idx: np.ndarray) -> dict:
    """
    V: (n, T, K) gradient norms (NaN = invalid candidate), own_idx: (n,) column of the stored
    label. Returns glare, glarex, margin (all: lower = more suspicious) and the best
    alternative column by glarex.
    """
    n, T, K = V.shape
    Vf = np.where(np.isnan(V), np.inf, V)
    arg = Vf.argmin(axis=2)                                   # first index on ties, like np.argmin
    onehot = np.eye(K, dtype=bool)[arg]                       # (n, T, K)
    wins = onehot.sum(axis=1).astype(float)                   # (n, K)
    norm_when_min = np.where(onehot, np.nan_to_num(V), 0.0).sum(axis=1)
    valid = ~np.isnan(V)
    norm_total = np.where(valid, V, 0.0).sum(axis=1)
    valid_total = valid.sum(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        avg = np.where(wins > 0, norm_when_min / np.maximum(wins, 1),
                       np.where(valid_total > 0, norm_total / np.maximum(valid_total, 1), np.nan))
        glarex = wins + np.where(avg > 0, 0.01 / avg, 0.0)
    glarex = np.where(np.isnan(avg), np.nan, glarex)

    r = np.arange(n)
    alt = np.where(np.isnan(glarex), -np.inf, glarex)
    alt[r, own_idx] = -np.inf
    alt_idx = alt.argmax(axis=1)

    own = V[r[:, None], np.arange(T)[None, :], own_idx[:, None]]
    others = Vf.copy()
    others[r, :, own_idx] = np.inf
    with np.errstate(divide="ignore", invalid="ignore"):
        margin = ((others.min(axis=2) - own) / own).mean(axis=1)

    return {"glare": wins[r, own_idx], "glarex": glarex[r, own_idx], "margin": margin,
            "alt_idx": alt_idx, "n_ckpt": T}


def checkpoint_names_from_info(info: dict, n: int) -> list[str]:
    """Checkpoint names in scoring order. Older merges (the 20-ep region run) store none;
    there the selection was 'all' over plain epoch ends, so index == epoch."""
    names = info.get("checkpoints") or (info.get("shards") or [{}])[0].get("checkpoints")
    names = [Path(c).name for c in names] if names else [f"epoch_{i}" for i in range(n)]
    if len(names) != n:
        raise ValueError(f"run_info lists {len(names)} checkpoints, grad_norms has {n}")
    return names


def select_checkpoints(V: np.ndarray, names: list[str], wanted: str | None):
    """Restrict V[:, T, K] to the named checkpoints (comma list, in training order)."""
    if not wanted:
        return V, names
    sel = [c.strip() for c in wanted.split(",") if c.strip()]
    missing = [c for c in sel if c not in names]
    if missing:
        raise SystemExit(f"checkpoints not in this run: {missing} (available: {names})")
    idx = sorted(names.index(c) for c in sel)
    return V[:, idx, :], [names[i] for i in idx]


def select_confidence(conf: pd.DataFrame, conf_dir, wanted: list[str]) -> pd.DataFrame:
    """Same checkpoint subset for the forward sweep, re-indexed 0..k-1 in training order."""
    infos = sorted(Path(conf_dir).glob("*confidence_info.json"))
    names = [Path(c).name for c in json.loads(infos[0].read_text())["checkpoints"]]
    missing = [c for c in wanted if c not in names]
    if missing:
        raise SystemExit(f"confidence sweep lacks checkpoints {missing}")
    remap = {names.index(c): i for i, c in enumerate(wanted)}
    conf = conf[conf["epoch"].isin(remap)].copy()
    conf["epoch"] = conf["epoch"].map(remap)
    return conf


def lr_per_checkpoint(spec: dict | None, names: list[str]) -> np.ndarray:
    """
    Learning rate of the epoch each checkpoint closes, for the TracIn weights eta_t.
    Mirrors DenseNetModel.configure_optimizers (Adam + LinearLR stepped per epoch, same
    get_parameter defaults). epoch_<n> and epoch_<n>_step_<s> both lie in epoch n.
    Without a spec all weights are 1 (plain sum of squared norms).
    """
    if spec is None:
        print("WARNING: no spec.json -- TracIn with eta_t = 1")
        return np.ones(len(names))
    from helper.arguments import get_parameter
    lr = get_parameter(spec, "lr", 1e-4, float)
    if not get_parameter(spec, "lr_scheduler", True, bool):
        return np.full(len(names), lr)
    end = get_parameter(spec, "lr_end_factor", 0.001, float)
    total = get_parameter(spec, "lr_total_iters", get_parameter(spec, "epochs", 25, int), int)
    ep = np.array([int(re.match(r"epoch_(\d+)", n).group(1)) for n in names], dtype=float)
    return lr * (1.0 + (end - 1.0) * np.minimum(ep, total) / total)


def conf_checkpoint_names(conf_dir) -> list[str]:
    infos = sorted(Path(conf_dir).glob("*confidence_info.json"))
    return [Path(c).name for c in json.loads(infos[0].read_text())["checkpoints"]]


def aum_scores(conf: pd.DataFrame, n_ckpt: int | None = None) -> pd.Series:
    """AUM per id: mean over checkpoints of the logit margin z_y - max_{j!=y} z_j."""
    if n_ckpt is not None:
        conf = conf[conf["epoch"] < n_ckpt]
    if "logits" in conf.columns:
        Z = np.stack(conf["logits"].to_numpy()).astype(np.float64)
    else:                     # log-softmax differs from the logits by a per-row constant
        Z = np.log(np.clip(np.stack(conf["p_all"].to_numpy()).astype(np.float64), 1e-45, None))
    y = conf["label"].to_numpy().astype(int)
    r = np.arange(len(Z))
    own = Z[r, y].copy()
    Z[r, y] = -np.inf
    return pd.Series(own - Z.max(axis=1), index=conf["id"].to_numpy()).groupby(level=0).mean()


def aum_alt_class(conf: pd.DataFrame) -> pd.Series:
    """AUM's correction label per id: argmax_{j!=y} of the checkpoint-mean logits (the class
    the margin is measured against, averaged over the trajectory)."""
    if "logits" in conf.columns:
        Z = np.stack(conf["logits"].to_numpy()).astype(np.float64)
    else:
        Z = np.log(np.clip(np.stack(conf["p_all"].to_numpy()).astype(np.float64), 1e-45, None))
    mean = pd.DataFrame(Z, index=conf["id"].to_numpy()).groupby(level=0).mean()
    y = conf.groupby("id")["label"].first().reindex(mean.index).to_numpy().astype(int)
    M = mean.to_numpy()
    M[np.arange(len(M)), y] = -np.inf
    return pd.Series(M.argmax(axis=1), index=mean.index)


def tracin_scores(V: np.ndarray, own_idx: np.ndarray, etas: np.ndarray) -> np.ndarray:
    """-sum_t eta_t ||g_t||^2 at the stored label; V (n, T, K) gradient norms."""
    T = V.shape[1]
    own = V[np.arange(len(V))[:, None], np.arange(T)[None, :], own_idx[:, None]]
    return -(own ** 2 * etas[None, :T]).sum(axis=1)


def load_vog_dir(vog_dir) -> pd.DataFrame:
    files = sorted(Path(vog_dir).glob("vog*.pkl"))
    if not files:
        raise FileNotFoundError(f"no vog*.pkl in {vog_dir} (run_vog.py run?)")
    vog = pd.concat([pd.read_pickle(f) for f in files], ignore_index=True)
    if vog["id"].duplicated().any():
        raise SystemExit(f"{vog_dir}: duplicate ids across vog*.pkl files")
    return vog


def vog_scores(vog: pd.DataFrame) -> pd.DataFrame:
    """Class-normalised VoG per id, negated so that lower = more suspicious (high VoG = hard)."""
    cols = [c for c in ("vog", "vog_early", "vog_late") if c in vog.columns]
    g = vog.groupby("label")[cols]
    sd = g.transform("std", ddof=0).replace(0.0, 1.0)
    z = (vog[cols] - g.transform("mean")) / sd
    out = -z
    out.index = vog["id"].to_numpy()
    return out


def confidence_scores(conf: pd.DataFrame, n_ckpt: int | None = None) -> pd.DataFrame:
    """Per id: conf_p (mean p_own) and conf_win (#checkpoints with pred == label)."""
    if n_ckpt is not None:
        conf = conf[conf["epoch"] < n_ckpt]
    g = conf.groupby("id")
    return pd.DataFrame({"conf_p": g["p_own"].mean(), "conf_win": g["correct"].sum()})


# ----------------------------------------------------------------------------------------
# Ground truth in the run's label space
# ----------------------------------------------------------------------------------------
VERT_NAME = {**{f"C{i}": i for i in range(1, 8)}, **{f"T{i}": 7 + i for i in range(1, 13)},
             **{f"L{i}": 19 + i for i in range(1, 6)}, "S1": 25, "T13": 28}


def true_vert(row) -> int:
    """True vertebra number for a GT row (for the vertebra space and suggestion checks)."""
    v = int(row["vert_id"])
    det = str(row.get("detail") or "")
    m = re.search(r"korrekt\s+([CTLS]\d+)", det)
    if m:
        return VERT_NAME[m.group(1)]
    t = row["anomaly_type"]
    if str(t).startswith("sim_"):
        return v        # synthetic (E2): the file number is the truth, only the label was changed
    if t == "T12_missing":
        return v + 1
    if t == "T13_extra":
        return 28 if v == 20 else v - 1
    raise ValueError(f"cannot derive true vertebra for {row['sample_id']} ({t}, {det!r})")


def true_label(row, space: str) -> int:
    tc = str(row["true_class"]).strip().lower()
    if space == "region":
        return REGION_NAMES.index(tc)
    if space == "region5":
        if row["anomaly_type"] == "T13_extra" and int(row["vert_id"]) == 20:
            return REGION5_NAMES.index("T13")
        return REGION5_NAMES.index(tc)
    tv = true_vert(row)                                          # vertebra: label = number - 1
    return tv - 1 if 1 <= tv <= 25 else -1


def load_gt(path: Path, space: str) -> pd.DataFrame:
    gt = pd.read_csv(path) if Path(path).suffix.lower() == ".csv" else pd.read_excel(path)
    gt["sample_id"] = gt["sample_id"].astype(str)
    gt["true_label"] = [true_label(r, space) for _, r in gt.iterrows()]
    return gt.set_index("sample_id")


# ----------------------------------------------------------------------------------------
# Metrics
# ----------------------------------------------------------------------------------------
def expected_tp_at(scores: np.ndarray, y: np.ndarray, frac: float) -> tuple[float, int]:
    """(expected TP, k) when the lowest `frac` of samples are checked; ties broken uniformly at random."""
    n = len(scores)
    k = int(round(frac * n))
    if y.sum() == 0 or k == 0:
        return 0.0, k
    s = np.sort(scores, kind="mergesort")
    cut = s[k - 1]
    below = scores < cut
    at = scores == cut
    need = k - below.sum()
    tp = y[below].sum() + (y[at].sum() * need / at.sum() if at.sum() else 0.0)
    return float(tp), k


def expected_recall_at(scores: np.ndarray, y: np.ndarray, frac: float) -> float:
    """Recall when the lowest `frac` of samples are checked; ties broken uniformly at random."""
    tp, _ = expected_tp_at(scores, y, frac)
    return tp / y.sum() if y.sum() else 0.0


def metrics_for(scores: np.ndarray, y: np.ndarray) -> dict:
    ok = ~np.isnan(scores)
    s, yy = scores[ok], y[ok]
    # +inf occurs in ll_margin when p_own == 1.0 in float32 (own last-layer norm 0 -> "maximally
    # clean"). sklearn rejects inf; every metric here is rank-based, so map +/-inf just beyond the
    # finite range (ties among them kept) -- the numbers are exactly those of the infinite scores.
    fin = np.isfinite(s)
    if not fin.all() and fin.any():
        hi, lo = s[fin].max(), s[fin].min()
        s = np.where(s == np.inf, hi + 1 + abs(hi), np.where(s == -np.inf, lo - 1 - abs(lo), s))
    out = {"n": int(ok.sum()), "positives": int(yy.sum())}
    if 0 < yy.sum() < len(yy):
        out["auroc"] = round(float(roc_auc_score(yy, -s)), 4)
        out["auprc"] = round(float(average_precision_score(yy, -s)), 4)
    for f in FRACTIONS:
        tp, k = expected_tp_at(s, yy, f)
        rec = tp / yy.sum() if yy.sum() else 0.0
        prec = tp / k if k else 0.0
        out[f"recall@{f * 100:g}%"] = round(100 * rec, 1)
        out[f"precision@{f * 100:g}%"] = round(100 * prec, 1)
        out[f"f1@{f * 100:g}%"] = round(100 * (2 * prec * rec / (prec + rec) if prec + rec else 0.0), 1)
    return out


def threshold_table(glare: np.ndarray, y: np.ndarray, n_ckpt: int) -> pd.DataFrame:
    rows = []
    for t in range(n_ckpt):
        pred = glare <= t
        tp, fp = int((pred & (y == 1)).sum()), int((pred & (y == 0)).sum())
        fn = int(y.sum()) - tp
        rec = tp / (tp + fn) if tp + fn else 0.0
        prec = tp / (tp + fp) if tp + fp else 0.0
        rows.append({"threshold": f"<= {t}", "flagged": tp + fp,
                     "fraction_checked_%": round(100 * (tp + fp) / len(y), 2),
                     "TP": tp, "recall_%": round(100 * rec, 1), "precision_%": round(100 * prec, 1),
                     "F1_%": round(100 * (2 * prec * rec / (prec + rec) if prec + rec else 0.0), 1)})
    return pd.DataFrame(rows)


def score_frame(ids, labels, V, space, offsets, conf=None, n_ckpt=None, etas=None,
                conf_etas=None, vog=None) -> pd.DataFrame:
    Vt = V if n_ckpt is None else V[:, :n_ckpt, :]
    own_idx = labels if offsets is None else np.full(len(ids), offsets.index(0))
    sc = glare_scores(Vt, own_idx)
    df = pd.DataFrame({"id": ids, "label": labels, "glare": sc["glare"], "glarex": sc["glarex"],
                       "margin": sc["margin"]})
    if etas is not None:
        df["tracin"] = tracin_scores(Vt, own_idx, etas)
    if offsets is None:
        df["alt_label"] = sc["alt_idx"]
    else:
        df["alt_label"] = labels + np.array(offsets)[sc["alt_idx"]]
    if conf is not None:
        c = confidence_scores(conf, n_ckpt)
        df = df.join(c, on="id")
        from helper.last_layer import last_layer_grad_norms
        ll = last_layer_grad_norms(conf if n_ckpt is None else conf[conf["epoch"] < n_ckpt])
        lcols = [c for c in ll.columns if c.endswith("_grad_norm")]
        lids, llab, LV = to_tensor(ll, lcols)
        lsc = glare_scores(LV, llab)
        ldf = pd.DataFrame({"id": lids, "ll_glarex": lsc["glarex"], "ll_margin": lsc["margin"]})
        if conf_etas is not None:
            ldf["tracin_ll"] = tracin_scores(LV, llab, conf_etas)
        df = df.merge(ldf, on="id", how="left")
        df = df.join(aum_scores(conf, n_ckpt).rename("aum"), on="id")
    if vog is not None:
        df = df.join(vog, on="id")
    return df


SCORERS = ("glare", "glarex", "margin", "conf_p", "conf_win", "ll_glarex", "ll_margin",
           "aum", "tracin", "tracin_ll", "vog", "vog_early", "vog_late")


def evaluate(df: pd.DataFrame, gt: pd.DataFrame, n_ckpt: int, dsname: pd.Series | None,
             exclude: set | None = None) -> dict:
    df = df[~df["id"].isin(exclude)] if exclude else df
    in_gt = df["id"].isin(gt.index)
    stored = df["label"].to_numpy()
    tl = df["id"].map(gt["true_label"]).to_numpy()
    # A GT row whose stored label already equals the true label (e.g. Rule-A T13 in region5)
    # is not a mislabel in this label space.
    resolved = in_gt.to_numpy() & (stored == tl)
    y = (in_gt.to_numpy() & ~resolved).astype(int)

    res = {"n_samples": int(len(df)), "n_gt_rows": int(len(gt)),
           "gt_matched": int(in_gt.sum()), "gt_resolved_by_label_space": int(resolved.sum()),
           "positives": int(y.sum()), "n_ckpt": n_ckpt, "scorers": {}}
    for s in SCORERS:
        if s in df.columns:
            res["scorers"][s] = metrics_for(df[s].to_numpy(dtype=float), y)
    strict = df["glare"].to_numpy() < n_ckpt
    res["strict"] = {"flagged": int(strict.sum()), "TP": int((strict & (y == 1)).sum()),
                     "recall_%": round(100 * (strict & (y == 1)).sum() / max(y.sum(), 1), 1),
                     "fraction_checked_%": round(100 * strict.mean(), 2)}

    # per anomaly type: strict, rank-based recall (glarex), suggestion correct
    rank = df["glarex"].rank(method="first").to_numpy() / len(df)
    d = df.assign(y=y, rank=rank, strict=strict, true_label=tl)
    pos = d[d["y"] == 1].join(gt[["anomaly_type", "dsname"]], on="id")
    per_type = []
    for t, g in pos.groupby("anomaly_type"):
        per_type.append({
            "anomaly_type": t, "n": len(g), "strict_found": int(g["strict"].sum()),
            **{f"found@{f * 100:g}%": int((g["rank"] <= f).sum()) for f in (0.01, 0.02, 0.05, 0.10)},
            "median_glare": float(g["glare"].median()),
            "suggestion_correct": int((g["alt_label"] == g["true_label"]).sum()),
            "suggestion_possible": int((g["true_label"] >= 0).sum()),
        })
    res["per_type"] = per_type

    if dsname is not None:
        d["dsname"] = d["id"].str.replace(r"_vert\d+$", "", regex=True).map(dsname)
        per_ds = []
        for ds, g in d.groupby("dsname"):
            per_ds.append({"dsname": ds, "n": len(g), "positives": int(g["y"].sum()),
                           "found@5%": int(((g["rank"] <= 0.05) & (g["y"] == 1)).sum()),
                           "flag_rate_strict_%": round(100 * g["strict"].mean(), 2),
                           "fpr_strict_%": round(100 * (g["strict"] & (g["y"] == 0)).sum()
                                                 / max((g["y"] == 0).sum(), 1), 2)})
        res["per_dsname"] = per_ds
    res["_frame"] = d
    return res


# ----------------------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--run-dir", required=True, help="GLARE run dir (contains grad_norms_*.pkl)")
    ap.add_argument("--gt", default=str(paths.GT_REGION_EXCEL))
    ap.add_argument("--sim-gt", default=None,
                    help="simulated GT (E2). Positives = simulated cases; real GT ids are "
                         "dropped from the evaluated population. Real-GT metrics are reported too.")
    ap.add_argument("--aum-dir", "--confidence-dir", dest="confidence_dir", default=None,
                    help="dir with confidence.pkl (run_aum.py) for the same run -> aum, tracin_ll, conf_*")
    ap.add_argument("--vog-dir", default=None,
                    help="dir with vog.pkl (run_vog.py) for the same run -> vog, vog_early, vog_late")
    ap.add_argument("--spec", default=None, help="training spec.json (default: from run_info.json)")
    ap.add_argument("--out-dir", default=None, help="default: <run-dir>/evaluation")
    ap.add_argument("--checkpoints", default=None,
                    help="score only these checkpoints (comma list of names), e.g. "
                         "epoch_0,epoch_1,epoch_2,epoch_3 = the control definition: the 20-epoch "
                         "schedule stopped after 4 epochs, no checkpoint before the end of epoch 0")
    ap.add_argument("--no-truncation", action="store_true")
    ap.add_argument("--tag", default="", help="free text stored in the output")
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    gn_path = newest(run_dir, "grad_norms_*.pkl")
    grad_norms = pd.read_pickle(gn_path)
    info = {}
    if (run_dir / "run_info.json").is_file():
        info = json.loads((run_dir / "run_info.json").read_text())
    spec = None
    spec_path = args.spec or info.get("spec_path")
    if not spec_path and info.get("weights_dir"):
        spec_path = str(Path(info["weights_dir"]).parent / "spec.json")
    if spec_path and Path(spec_path).is_file():
        spec = json.loads(Path(spec_path).read_text())

    space, cols, offsets = detect_space(grad_norms, spec)
    ids, labels, V = to_tensor(grad_norms, cols)
    ckpt_names = checkpoint_names_from_info(info, V.shape[1])
    V, ckpt_names = select_checkpoints(V, ckpt_names, args.checkpoints)
    n_ckpt = V.shape[1]
    print(f"{run_dir.name}: {len(ids)} samples x {n_ckpt} checkpoints ({ckpt_names[0]} ... "
          f"{ckpt_names[-1]}), label space {space}")

    conf = None
    if args.confidence_dir:
        from helper.last_layer import load_confidence_dir
        conf = load_confidence_dir(args.confidence_dir)
        if args.checkpoints:
            conf = select_confidence(conf, args.confidence_dir, ckpt_names)
        if conf["epoch"].nunique() != n_ckpt:
            print(f"WARNING: confidence has {conf['epoch'].nunique()} checkpoints, GLARE {n_ckpt}; "
                  "truncation compares the first N of each.")
        conf_names = ckpt_names if args.checkpoints else conf_checkpoint_names(args.confidence_dir)
        conf_etas = lr_per_checkpoint(spec, conf_names)
    else:
        conf_etas = None
    etas = lr_per_checkpoint(spec, ckpt_names)
    print("TracIn eta_t: " + ", ".join(f"{n}={e:.3g}" for n, e in zip(ckpt_names, etas)))

    vog = None
    if args.vog_dir:
        vraw = load_vog_dir(args.vog_dir)
        lab = pd.Series(labels, index=ids)
        common = vraw["id"].isin(lab.index)
        if (~common).any() or len(vraw) != len(ids):
            print(f"WARNING: VoG covers {int(common.sum())} of {len(ids)} GLARE samples "
                  f"({int((~common).sum())} unknown ids)")
        bad = vraw[common & (vraw["id"].map(lab) != vraw["label"])]
        if len(bad):
            raise SystemExit(f"VoG and GLARE disagree on {len(bad)} stored labels (other run/overrides?)")
        vog = vog_scores(vraw[common])

    try:
        filt = pd.read_excel(paths.EXCEL_FILTER, usecols=["pid", "dsname"])
        dsname = filt.drop_duplicates("pid").set_index("pid")["dsname"]
    except Exception as e:                                         # pragma: no cover
        print(f"no dsname breakdown ({e})")
        dsname = None

    gt_real = load_gt(Path(args.gt), space)
    gts = {"real": (gt_real, None)}
    if args.sim_gt:
        gts = {"sim": (load_gt(Path(args.sim_gt), space), set(gt_real.index)), "real": (gt_real, None)}

    full = score_frame(ids, labels, V, space, offsets, conf, etas=etas, conf_etas=conf_etas, vog=vog)
    out = {"run_dir": str(run_dir), "grad_norms": gn_path.name, "label_space": space,
           "n_ckpt": n_ckpt, "checkpoints": ckpt_names, "checkpoint_selection": args.checkpoints,
           "tag": args.tag,
           "gt": args.gt, "sim_gt": args.sim_gt, "confidence_dir": args.confidence_dir,
           "vog_dir": args.vog_dir, "tracin_eta": dict(zip(ckpt_names, etas.tolist())),
           "timestamp": datetime.now().isoformat(), "results": {}}
    frames = {}
    for name, (gt, excl) in gts.items():
        r = evaluate(full, gt, n_ckpt, dsname, excl)
        frames[name] = r.pop("_frame")
        r["threshold_table"] = threshold_table(frames[name]["glare"].to_numpy(),
                                               frames[name]["y"].to_numpy(), n_ckpt).to_dict("records")
        if not args.no_truncation:
            trunc = []
            for N in range(1, n_ckpt + 1):
                tdf = score_frame(ids, labels, V, space, offsets, conf, n_ckpt=N, etas=etas,
                                  conf_etas=conf_etas)
                tr = evaluate(tdf, gt, N, None, excl)
                tr.pop("_frame")
                for s, m in tr["scorers"].items():
                    trunc.append({"N": N, "scorer": s, **{k: v for k, v in m.items()
                                                           if k not in ("n", "positives")}})
            r["truncation"] = trunc
        out["results"][name] = r

    out_dir = Path(args.out_dir) if args.out_dir else run_dir / "evaluation"
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    (out_dir / f"evaluation_{ts}.json").write_text(json.dumps(out, indent=2, default=float))
    with pd.ExcelWriter(out_dir / f"evaluation_{ts}.xlsx", engine="openpyxl") as xw:
        for name, r in out["results"].items():
            pd.DataFrame([{"scorer": s, **m} for s, m in r["scorers"].items()]).to_excel(
                xw, sheet_name=f"{name}_scorers", index=False)
            pd.DataFrame(r["per_type"]).to_excel(xw, sheet_name=f"{name}_per_type", index=False)
            pd.DataFrame(r["threshold_table"]).to_excel(xw, sheet_name=f"{name}_thresholds", index=False)
            if "per_dsname" in r:
                pd.DataFrame(r["per_dsname"]).to_excel(xw, sheet_name=f"{name}_dsname", index=False)
            if "truncation" in r:
                pd.DataFrame(r["truncation"]).to_excel(xw, sheet_name=f"{name}_truncation", index=False)
            frames[name].sort_values("glarex").to_excel(xw, sheet_name=f"{name}_ranked", index=False)

    for name, r in out["results"].items():
        print(f"\n== {name} GT: {r['positives']} positives ({r['gt_matched']}/{r['n_gt_rows']} matched, "
              f"{r['gt_resolved_by_label_space']} resolved by label space)")
        print(f"strict (score < {n_ckpt}): {r['strict']}")
        print(pd.DataFrame([{"scorer": s, **m} for s, m in r["scorers"].items()]).to_string(index=False))
        print(pd.DataFrame(r["per_type"]).to_string(index=False))
    print(f"\nwritten: {out_dir}/evaluation_{ts}.json / .xlsx")


if __name__ == "__main__":
    main()
