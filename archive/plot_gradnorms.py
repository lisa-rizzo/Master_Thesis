# Usage:
#   cd /home/student/lisa_ma
#   PYTHONPATH=. python figures/plot_gradnorms.py /home/student/lisa_ma/results/glare/08-10-25_21:28_ep-15_single-gpu

import sys
from pathlib import Path
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Try to import helper functions if available, otherwise provide local fallbacks.
try:
    from helper.grad_norms_dict import get_n_epoch, dict2numpy
except Exception:
    # Fallback implementations that work with your grad_norms.pkl DataFrame format.
    def get_n_epoch(grad_obj):
        """
        grad_obj may be a pandas.DataFrame with column 'epoch' or a dict-like structure.
        Returns number of unique epochs.
        """
        if isinstance(grad_obj, pd.DataFrame):
            return int(grad_obj['epoch'].max() + 1)
        # generic fallback
        try:
            keys = list(grad_obj.keys())
            # if dict-of-epochs, guess length
            return len(keys)
        except Exception:
            raise RuntimeError("get_n_epoch fallback: Unknown grad_obj structure")

    def dict2numpy(grad_obj, key="grad_norm"):
        """
        Convert grad_obj into (all_samples, indices).
        For your DataFrame format (id, epoch, c0_grad_norm, c1_grad_norm...), compute
        a single combined grad-norm per row (Euclidean across class columns), then pivot to
        shape (n_samples, n_epochs) and return (numpy array, index array-of-ids).
        """
        if isinstance(grad_obj, pd.DataFrame):
            df = grad_obj.copy()
            # detect grad columns like c0_grad_norm, c1_grad_norm, ...
            grad_cols = [c for c in df.columns if c.endswith("_grad_norm")]
            if len(grad_cols) == 0:
                raise RuntimeError("dict2numpy fallback: keine '*_grad_norm' Spalten gefunden.")

            # convert to numeric
            df[grad_cols] = df[grad_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)

            # combine per-row: Euclidean norm across class columns (you can change to sum/abs if you prefer)
            df["_combined_grad_norm"] = np.sqrt((df[grad_cols].values ** 2).sum(axis=1))

            # pivot to samples x epochs
            pivot = df.pivot_table(index="id", columns="epoch", values="_combined_grad_norm", aggfunc="mean")

            # sort columns by epoch index (in case)
            pivot = pivot.reindex(sorted(pivot.columns), axis=1)

            # fill missing epochs with np.nan (keeps shape consistent)
            all_samples = pivot.values.astype(float)  # shape: (n_samples, n_epochs)
            indices = pivot.index.astype(str).values  # sample ids
            return all_samples, indices

        # if it's already a dict-like expected by older helper, try to handle minimal cases:
        if isinstance(grad_obj, dict):
            # try: keys might be ids each mapping to per-epoch arrays
            try:
                ids = list(grad_obj.keys())
                sample0 = grad_obj[ids[0]]
                arr0 = np.asarray(sample0)
                # assume dict: id -> array(len_epochs)
                mat = np.vstack([np.asarray(grad_obj[i]) for i in ids])
                return mat, np.array(ids, dtype=str)
            except Exception as e:
                raise RuntimeError(f"dict2numpy fallback could not convert dict: {e}")

        raise RuntimeError("dict2numpy fallback: Unrecognized grad_obj type")

# --- keep colleague function unchanged (just use above dict2numpy/get_n_epoch) ---
def make_figure_metricavg_over_epochs(
    gradnorms_dict: dict,
    diff_indices_i: list[int],
    key: str = "grad_norm",
    take_absolute: bool = False,
):
    n_epoch = get_n_epoch(gradnorms_dict)
    # transforms the dict into a 2D numpy array for that key
    all_samples, indices = dict2numpy(gradnorms_dict, key)

    # Boolean masks for subsets
    mis_mask = np.isin(indices, list(diff_indices_i))
    clean_mask = ~mis_mask

    # take absolute
    if take_absolute:
        all_samples = np.abs(all_samples)

    # Medians
    all_median = np.nanmedian(all_samples, axis=0)
    mis_median = np.nanmedian(all_samples[mis_mask], axis=0) if np.any(mis_mask) else np.full(all_median.shape, np.nan)
    clean_median = np.nanmedian(all_samples[clean_mask], axis=0) if np.any(clean_mask) else np.full(all_median.shape, np.nan)

    # Standard deviations (ignore NaNs)
    all_std = np.nanstd(all_samples, axis=0)
    mis_std = np.nanstd(all_samples[mis_mask], axis=0) if np.any(mis_mask) else np.zeros_like(all_std)
    clean_std = np.nanstd(all_samples[clean_mask], axis=0) if np.any(clean_mask) else np.zeros_like(all_std)

    # Plot
    fig, ax = plt.subplots(figsize=(9,5))

    epochs = np.arange(n_epoch)

    # All
    ax.plot(epochs, all_median, color="blue", label="all")
    ax.fill_between(epochs, all_median - all_std, all_median + all_std, color="blue", alpha=0.2)

    # Mislabel
    ax.plot(epochs, mis_median, color="red", label="mislabel")
    ax.fill_between(epochs, mis_median - mis_std, mis_median + mis_std, color="red", alpha=0.2)

    # Clean
    ax.plot(epochs, clean_median, color="green", label="clean")
    ax.fill_between(epochs, clean_median - clean_std, clean_median + clean_std, color="green", alpha=0.2)
    ax.legend()
    ax.set_xlabel("Epoch")
    ax.set_ylabel(f"Median {key}")
    ax.set_title("Grad-norms over epochs (median ± std)")
    ax.grid(alpha=0.25)
    plt.tight_layout()
    return fig

def detect_and_map_diff_indices(grad_dict, mis_ids, results_dir):
    """
    Returns diff_indices_i in the same format used by dict2numpy indices:
    - if dict2numpy returns integer positions, returns position indices
    - if dict2numpy returns string IDs, returns those IDs (strings)
    """
    # Try to get arrays for key "grad_norm" (exists in your run)
    try:
        all_samples, indices = dict2numpy(grad_dict, key="grad_norm")
    except Exception as e:
        raise RuntimeError(f"Fehler beim Aufruf von dict2numpy: {e}")

    indices = np.array(indices)
    # If indices look integer-like (dtype int or numeric strings), treat as positions
    if np.issubdtype(indices.dtype, np.integer):
        # indices are positions (0..N-1) -> we need to map mis_ids (strings) -> positions
        scores_pkl = Path(results_dir) / "scores.pkl"
        if not scores_pkl.exists():
            print("[WARN] indices sind Integer, aber scores.pkl nicht gefunden; versuche numerische Mis-IDs")
            try:
                pos_list = [int(x) for x in mis_ids]
                return pos_list
            except Exception:
                raise RuntimeError("Kann mislabels nicht zu Positionen mapen: keine scores.pkl und mis_ids sind keine Integer.")
        scores = pd.read_pickle(scores_pkl)
        if 'id' not in scores.columns:
            raise RuntimeError("scores.pkl gefunden, aber enthält keine 'id' Spalte zum Mapping.")
        id_list = list(scores['id'].astype(str).values)
        id_to_pos = {id_: pos for pos, id_ in enumerate(id_list)}
        pos = [id_to_pos[x] for x in mis_ids if x in id_to_pos]
        if len(pos) == 0:
            raise RuntimeError("Keine Übereinstimmung zwischen mislabels IDs und scores.pkl IDs gefunden.")
        return pos

    else:
        # indices look like strings/IDs -> we can use mis_ids directly (filter to intersection)
        indices_str = indices.astype(str)
        mis_filtered = [x for x in mis_ids if str(x) in set(indices_str)]
        if len(mis_filtered) == 0:
            print("[WARN] Keine Übereinstimmung zwischen mislabels IDs und grad_norms indices; returning all mis IDs (may be empty).")
        return mis_filtered

def main(argv):
    # default result dir (from your run)
    default_dir = Path("results/glare/08-10-25_21:28_ep-15_single-gpu")
    results_dir = Path(argv[1]) if len(argv) > 1 else default_dir
    if not results_dir.exists():
        print("Result directory nicht gefunden:", results_dir)
        return

    grad_pkl = results_dir / "grad_norms.pkl"
    mis_csv = results_dir / "mislabels_only.csv"

    if not grad_pkl.exists():
        print("grad_norms.pkl nicht gefunden in", results_dir)
        return
    if not mis_csv.exists():
        print("mislabels_only.csv nicht gefunden in", results_dir)
        return

    # load
    print("Lade grad_norms.pkl ...")
    grad = pickle.load(open(grad_pkl, "rb"))
    print("Lade mislabels ...")
    mis = pd.read_csv(mis_csv)
    mis_ids = list(mis['id'].astype(str).values)
    print(f"Gefundene mislabels: {len(mis_ids)}")

    # map to indices used by dict2numpy
    try:
        diff_indices_i = detect_and_map_diff_indices(grad, mis_ids, results_dir)
    except RuntimeError as e:
        print("Mapping Fehler:", e)
        return

    print("diff_indices sample (first 20):", diff_indices_i[:20])

    # create output folder
    fig_dir = results_dir / "figures"
    fig_dir.mkdir(exist_ok=True)

    # call colleague's function (unchanged)
    fig = make_figure_metricavg_over_epochs(grad, diff_indices_i, key="grad_norm", take_absolute=False)
    out_file = fig_dir / "gradnorms_over_epochs_colleague_style.png"
    fig.savefig(out_file, dpi=200)
    plt.close(fig)
    print("Saved:", out_file)

    # also produce absolute-value version (optional)
    fig2 = make_figure_metricavg_over_epochs(grad, diff_indices_i, key="grad_norm", take_absolute=True)
    out_file2 = fig_dir / "gradnorms_over_epochs_colleague_style_abs.png"
    fig2.savefig(out_file2, dpi=200)
    plt.close(fig2)
    print("Saved:", out_file2)

if __name__ == "__main__":
    main(sys.argv)