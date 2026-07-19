import argparse
from pathlib import Path
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# optional visuals
try:
    import seaborn as sns
except Exception:
    sns = None

def safe_mkdir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def load_objs(run_dir: Path):
    objs = {}
    scores_pkl = run_dir / "scores.pkl"
    grad_pkl = run_dir / "grad_norms.pkl"
    mis_list = run_dir / "mislabels_list.csv"
    mis_only = run_dir / "mislabels_only.csv"
    if not scores_pkl.exists():
        raise FileNotFoundError(f"Missing {scores_pkl}")
    objs['scores'] = pd.read_pickle(scores_pkl)
    if grad_pkl.exists():
        objs['grad'] = pd.read_pickle(grad_pkl)
    else:
        objs['grad'] = None
    if mis_only.exists():
        objs['mis_csv'] = pd.read_csv(mis_only)
    elif mis_list.exists():
        objs['mis_csv'] = pd.read_csv(mis_list)
    else:
        objs['mis_csv'] = None
    return objs

def plot_glarex_distribution(s, out_dir: Path):
    fig, ax = plt.subplots(figsize=(8,4))
    if sns:
        sns.histplot(s['glarex score'], bins=100, kde=True, stat='density', color='C0', ax=ax)
    else:
        ax.hist(s['glarex score'], bins=100, color='C0', density=True)
    q10 = s['glarex score'].quantile(0.1)
    ax.axvline(q10, color='red', linestyle='--', label='10% quantile')
    ax.set_xlabel('glarex score'); ax.set_ylabel('density'); ax.set_title('Glarex distribution')
    ax.legend()
    fig.tight_layout()
    p = out_dir / "glarex_distribution.png"
    fig.savefig(p, dpi=200)
    plt.close(fig)
    return p

def plot_cdf(s, out_dir: Path):
    vals = np.sort(s['glarex score'].values)
    cdf = np.arange(1, len(vals)+1) / len(vals)
    fig, ax = plt.subplots(figsize=(6,4))
    ax.plot(vals, cdf)
    ax.set_xlabel('glarex score'); ax.set_ylabel('CDF'); ax.grid(alpha=0.3)
    p = out_dir / "glarex_cdf.png"
    fig.tight_layout(); fig.savefig(p, dpi=200); plt.close(fig)
    return p

def top_lists(s, out_dir: Path):
    s = s.copy()
    s['delta_glarex'] = s['alternative glarex score'] - s['glarex score']
    top200 = s.sort_values('delta_glarex', ascending=False).head(200)
    top200.to_csv(out_dir / "top200_by_delta_glarex.csv", index=False)
    top20 = top200.head(20)
    top20.to_csv(out_dir / "top20_by_delta_glarex.csv", index=False)
    (s[s['alternative glare score'] > s['glare score']].sort_values('alternative glare score', ascending=False)
        .head(200).to_csv(out_dir / "top_altcount_gt_label_head200.csv", index=False))
    (s[s['glare score'] <= 3].sort_values('glare score').to_csv(out_dir / "glare_le_3.csv", index=False))
    return top20, top200

def plot_delta_hist(s, out_dir: Path):
    fig, ax = plt.subplots(figsize=(8,4))
    if 'delta_glarex' not in s.columns:
        s['delta_glarex'] = s['alternative glarex score'] - s['glarex score']
    ax.hist(s['delta_glarex'], bins=100, color='C1')
    ax.axvline(0, color='k', linestyle='--')
    ax.set_xlabel('delta (alt_glarex - label_glarex)'); ax.set_ylabel('count'); ax.set_title('Delta glarex distribution')
    p = out_dir / "delta_glarex_hist.png"
    fig.tight_layout(); fig.savefig(p, dpi=200); plt.close(fig)
    return p

def plot_label_to_alt_heatmap(mis_df, out_dir: Path):
    import matplotlib.pyplot as plt
    try:
        import seaborn as sns
    except:
        sns = None
    if mis_df is None:
        return None
    mat = pd.crosstab(mis_df['label'], mis_df['alternative class (glare)'])
    fig, ax = plt.subplots(figsize=(6,5))
    if sns:
        sns.heatmap(mat, annot=True, fmt='d', cmap='Blues', ax=ax)
    else:
        im = ax.imshow(mat.values, cmap='Blues', aspect='auto')
        ax.set_xticks(np.arange(mat.shape[1])); ax.set_yticks(np.arange(mat.shape[0]))
        ax.set_xticklabels(mat.columns); ax.set_yticklabels(mat.index)
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                ax.text(j, i, str(mat.iat[i,j]), ha='center', va='center', color='k')
        fig.colorbar(im, ax=ax)
    ax.set_xlabel('alternative class (glare)'); ax.set_ylabel('label'); ax.set_title('label -> alternative (counts)')
    p = out_dir / "label_to_alt_heatmap.png"
    fig.tight_layout(); fig.savefig(p, dpi=200); plt.close(fig)
    return p

def plot_box_by_label(s, out_dir: Path):
    fig, ax = plt.subplots(figsize=(7,4))
    if sns:
        sns.boxplot(x='label', y='glarex score', data=s, ax=ax)
    else:
        groups = [grp['glarex score'].values for _,grp in s.groupby('label')]
        ax.boxplot(groups, labels=[str(g) for g,_ in s.groupby('label')])
        ax.set_xlabel('label'); ax.set_ylabel('glarex score')
    ax.set_title('glarex per label')
    p = out_dir / "glarex_by_label_boxplot.png"
    fig.tight_layout(); fig.savefig(p, dpi=200); plt.close(fig)
    return p

def gradnorms_mis_vs_clean(g_df, mis_ids, out_dir: Path):
    if g_df is None:
        return None
    grad_cols = [c for c in g_df.columns if c.endswith('_grad_norm')]
    g_df[grad_cols] = g_df[grad_cols].apply(pd.to_numeric, errors='coerce').fillna(0)
    g_df['_combined'] = np.sqrt((g_df[grad_cols].values**2).sum(axis=1))
    pivot = g_df.pivot_table(index='id', columns='epoch', values='_combined', aggfunc='mean')
    if mis_ids:
        mis_set = set(mis_ids)
        mis_df = pivot.loc[[i for i in pivot.index if i in mis_set]] if any(i in mis_set for i in pivot.index) else pivot.sample(min(200, len(pivot)), random_state=1)
    else:
        mis_df = pivot.sample(min(200, len(pivot)), random_state=1)
    clean_df = pivot.drop(mis_df.index, errors='ignore') if len(mis_df) < len(pivot) else pivot.iloc[:0]
    median_mis = mis_df.median(axis=0)
    median_clean = clean_df.median(axis=0) if len(clean_df)>0 else pivot.median(axis=0)
    epochs = median_clean.index.values
    fig, ax = plt.subplots(figsize=(8,4))
    ax.plot(epochs, median_clean, label='clean median', color='green')
    ax.plot(epochs, median_mis, label='mis median', color='red')
    ax.set_xlabel('epoch'); ax.set_ylabel('combined grad-norm (median)'); ax.legend(); ax.set_title('Grad-norms: mis vs clean median over epochs')
    p = out_dir / "gradnorms_mis_vs_clean_median.png"
    fig.tight_layout(); fig.savefig(p, dpi=200); plt.close(fig)
    return p

def find_npz_for_id(id_str: str, data_root: Path):
    if "_vert" not in id_str:
        return None
    pid_part, vert_part = id_str.rsplit("_vert", 1)
    vert_tok = f"vert{vert_part}"
    # naive search; could be optimized
    for p in data_root.rglob("*.npz"):
        if f"_sub-{pid_part}" in p.name and vert_tok in p.name:
            return p
    return None

def render_thumbnails(id_list, out_dir: Path, data_root: Path, max_count=48):
    out_dir.mkdir(parents=True, exist_ok=True)
    ids = id_list[:max_count]
    saved = []
    for i, id_ in enumerate(ids):
        npz = find_npz_for_id(id_, data_root)
        if npz is None:
            print("not found:", id_)
            continue
        data = np.load(npz)
        img = data['img']
        z = img.shape[0]//2
        axial = img[z,:,:]
        cor = np.max(img, axis=2)
        fig, ax = plt.subplots(1,2,figsize=(6,3))
        ax[0].imshow(axial, cmap='gray'); ax[0].axis('off'); ax[0].set_title('axial')
        ax[1].imshow(cor, cmap='gray'); ax[1].axis('off'); ax[1].set_title('coronal MIP')
        plt.suptitle(id_, fontsize=8)
        fp = out_dir / f"thumb_{i:03d}.png"
        fig.tight_layout(); fig.savefig(fp, dpi=150, bbox_inches='tight'); plt.close(fig)
        saved.append(fp)
    return saved

def main():
    parser = argparse.ArgumentParser(description="GLARE quick QC & plots")
    parser.add_argument("--run-dir", "-r", default=os.environ.get("RUN_DIR"), help="Run folder (contains scores.pkl, grad_norms.pkl, ...)")
    parser.add_argument("--data-root", "-d", default="prepared", help="Root folder where .npz files are stored (used for thumbnails)")
    parser.add_argument("--thumbs", action="store_true", help="Render thumbnails for top candidates")
    parser.add_argument("--top-k", type=int, default=48, help="How many thumbs to render")
    parser.add_argument("--no-seaborn", action="store_true", help="Disable seaborn even if available")
    args = parser.parse_args()

    if args.run_dir is None:
        print("ERROR: --run-dir or RUN_DIR env var must be set", file=sys.stderr)
        sys.exit(2)
    run_dir = Path(args.run_dir)
    if not run_dir.exists():
        print("ERROR: run_dir does not exist:", run_dir, file=sys.stderr)
        sys.exit(2)
    figures = run_dir / "figures"
    safe_mkdir(figures)

    objs = load_objs(run_dir)
    scores = objs['scores']
    grad = objs['grad']
    mis_df = objs['mis_csv']

    # Save a short textual summary
    summary_lines = []
    summary_lines.append(f"Run dir: {run_dir}")
    summary_lines.append(f"Total samples (scores.pkl): {len(scores)}")
    if grad is not None:
        summary_lines.append(f"grad_norms rows: {len(grad)}, unique ids: {grad['id'].nunique()}, epochs: {grad['epoch'].nunique()}")
    if mis_df is not None:
        summary_lines.append(f"mislabels rows: {len(mis_df)}")
    # basic stats
    gs = scores['glare score'].value_counts().sort_index()
    summary_lines.append("glare score distribution:")
    summary_lines += [f"  {int(k)}: {int(v)}" for k,v in gs.items()]
    summary_lines.append("glarex score summary:")
    summary_lines += list(map(str, scores['glarex score'].describe().to_dict().items()))
    # write summary file
    (run_dir / "checks_summary.txt").write_text("\n".join(summary_lines))

    # Plots and lists
    print("Creating glarex distribution plot...")
    p1 = plot_glarex_distribution(scores, figures)
    print("Creating glarex CDF...")
    p2 = plot_cdf(scores, figures)
    print("Creating delta distribution and top lists...")
    p_delta = plot_delta_hist(scores, figures)
    top20, top200 = top_lists(scores, figures)
    print("Creating label->alt heatmap (if mislabels present)...")
    hm = plot_label_to_alt_heatmap(mis_df, figures)
    print("Creating boxplot per label...")
    p_box = plot_box_by_label(scores, figures)
    print("Creating gradnorms comparison (if available)...")
    gm = gradnorms_mis_vs_clean(grad, list(mis_df['id'].astype(str)) if mis_df is not None else [], figures)

    # thumbnails
    thumbs_out = run_dir / "qc_thumbs"
    if args.thumbs:
        print(f"Rendering thumbnails for top {args.top_k} candidates...")
        # use top200_by_delta_glarex.csv if exists
        candidate_ids = None
        t200 = figures / "top200_by_delta_glarex.csv"
        if t200.exists():
            candidate_ids = pd.read_csv(t200)['id'].astype(str).tolist()
        else:
            candidate_ids = list(top200['id'].astype(str)) if top200 is not None else []
        saved = render_thumbnails(candidate_ids, thumbs_out, Path(args.data_root), max_count=args.top_k)
        print(f"Saved {len(saved)} thumbnails to {thumbs_out}")
    else:
        print("Thumbnails disabled (--thumbs to enable)")

    print("Done. Outputs saved to:", figures)
    print("Summary file:", run_dir / "checks_summary.txt")
    print("Top lists: ", figures / "top20_by_delta_glarex.csv", figures / "top200_by_delta_glarex.csv")

if __name__ == "__main__":
    main()