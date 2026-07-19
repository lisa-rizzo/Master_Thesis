#!/usr/bin/env python3
"""
Minimal script: print total samples, threshold (quantile), glare-score distribution and glarex summary.

Usage:
  python helper/compute_threshold_stats.py --scores path/to/scores.pkl
  python helper/compute_threshold_stats.py --scores path/to/scores.pkl --save out.txt
  python helper/compute_threshold_stats.py --scores path/to/scores.pkl --save-json out.json

Options:
  --scores     required, path to scores.pkl or scores.csv
  --method     optional, base name of score column (default: glarex)
  --q          optional, quantile fraction (default: 0.1)
  --save       optional, path to write the textual output (plain text)
  --save-json  optional, path to write a structured JSON summary
"""
import argparse
from pathlib import Path
import pandas as pd
import sys
import json

def load_scores(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"Scores file not found: {path}")
    if path.suffix in (".pkl", ".pickle"):
        return pd.read_pickle(path)
    return pd.read_csv(path)

def series_to_plain(s: pd.Series) -> str:
    """Format pandas Series (describe) to readable multiline text."""
    return s.to_string()

def dist_to_text(dist: pd.Series) -> str:
    return dist.to_string()

def build_text_output(total, method_col, q, thr, dist_text, glarex_desc_text):
    parts = []
    parts.append(f"Total samples: {total}")
    parts.append("")
    parts.append(f"Threshold ({q*100:.0f}%-Quantil) on '{method_col}': {thr:.6f}")
    parts.append("")
    parts.append("Glare score distribution (count per unique glare score):")
    parts.append(dist_text)
    parts.append("")
    parts.append(f"{method_col} summary (describe):")
    parts.append(glarex_desc_text)
    return "\n".join(parts)

def build_json_output(total, method_col, q, thr, dist: pd.Series, glarex_desc: pd.Series):
    # Convert dist and describe to simple serializable forms
    return {
        "total_samples": int(total),
        "threshold": {"method_column": method_col, "quantile": float(q), "threshold_value": float(thr)},
        "glare_score_distribution": dist.astype(int).to_dict(),
        "glarex_summary": glarex_desc.to_dict()
    }

def main():
    p = argparse.ArgumentParser(description="Minimal GLARE threshold + stats (with optional save)")
    p.add_argument("--scores", required=True, help="scores.pkl or scores.csv")
    p.add_argument("--method", default="glarex", help="method base name (default: glarex)")
    p.add_argument("--q", type=float, default=0.1, help="quantile fraction for threshold (default: 0.1)")
    p.add_argument("--save", help="optional path to save textual output")
    p.add_argument("--save-json", help="optional path to save structured JSON output")
    args = p.parse_args()

    scores_path = Path(args.scores)
    try:
        scores = load_scores(scores_path)
    except Exception as e:
        print(f"ERROR loading scores: {e}", file=sys.stderr)
        sys.exit(2)

    total = len(scores)
    method_col = f"{args.method} score"
    if method_col not in scores.columns:
        print(f"ERROR: expected column '{method_col}' not found. Available columns: {list(scores.columns)}", file=sys.stderr)
        sys.exit(2)

    thr = scores[method_col].quantile(args.q)

    # glare score distribution
    glare_col = "glare score"
    if glare_col in scores.columns:
        dist = scores[glare_col].value_counts().sort_index()
        dist_text = dist_to_text(dist)
    else:
        dist = pd.Series(dtype=int)
        dist_text = f"Warning: column '{glare_col}' not found; cannot compute glare score distribution."

    # glarex summary
    glarex_desc = scores[method_col].describe()
    glarex_desc_text = series_to_plain(glarex_desc)

    # build textual output
    text_out = build_text_output(total=total, method_col=method_col, q=args.q, thr=thr,
                                 dist_text=dist_text, glarex_desc_text=glarex_desc_text)

    # print to console
    print(text_out)

    # optionally save text
    if args.save:
        save_path = Path(args.save)
        try:
            save_path.parent.mkdir(parents=True, exist_ok=True)
            save_path.write_text(text_out, encoding="utf-8")
            print(f"\nTextual output written to: {save_path}")
        except Exception as e:
            print(f"ERROR writing text output to {save_path}: {e}", file=sys.stderr)

    # optionally save JSON
    if args.save_json:
        json_path = Path(args.save_json)
        try:
            json_obj = build_json_output(total=total, method_col=method_col, q=args.q, thr=thr,
                                         dist=dist, glarex_desc=glarex_desc)
            json_path.parent.mkdir(parents=True, exist_ok=True)
            with json_path.open("w", encoding="utf-8") as f:
                json.dump(json_obj, f, indent=2, ensure_ascii=False)
            print(f"JSON summary written to: {json_path}")
        except Exception as e:
            print(f"ERROR writing JSON output to {json_path}: {e}", file=sys.stderr)

if __name__ == "__main__":
    main()