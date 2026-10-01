# Mislabel Detection in 3D Spine CT

---

## Setup

```bash
uv sync                    # Python 3.11, PyTorch cu126 — see pyproject.toml / uv.lock
export MISLABELDET_DATASET_DIR=/path/to/prepared   # CT crops (74 GB, not in repo)
```

If `uv` is not installed: `curl -LsSf https://astral.sh/uv/install.sh | sh`

CT crops are needed only for the two example-image figures. All other figures and all evaluation scripts run without them.

---

## 1. Verify the thesis figures (no GPU needed)

Pre-computed run outputs are included in `baselines_data/data/` (see manifest below).

```bash
jupyter lab thesis_figures.ipynb   # "Run All", ~8 min CPU
# Writes all 23 figures to figures/
```

---

## 2. Train the classifier

```bash
# With LabelOverride (main run):
uv run python train_VerSe_densenet.py --spec configs/spec_lo_example.json

# Without LabelOverride (upper bound):
uv run python train_VerSe_upper_bound.py --spec configs/spec_lo_example.json
```

---

## 3. GLARE

```bash
uv run python run_glare.py \
    --spec configs/spec_lo_example.json \
    --weights-dir results/VerSe_classifier/label_override_8ep_23092026_2135/weights \
    --out-dir <out>
```

---

## 4. Baselines (AUM / VoG / TracIn)

| Baseline | Script | Score |
|---|---|---|
| **AUM** (Pleiss et al. 2020) | `run_aum.py` | mean margin `z_y − max_{j≠y} z_j` |
| **VoG** (Agarwal et al. 2022) | `run_vog.py` | −(z-score of mean-voxel std of `∂z_y/∂x` across checkpoints) |
| **TracIn** (Pruthi et al. 2020) | `evaluation/evaluate_run.py` | `−Σ_t η_t ‖∇_θ ℓ(w_t, z, y)‖²` — uses GLARE grad norms |
| **TracIn-LL** | `evaluation/evaluate_run.py` | same, closed form on classifier head — uses AUM output |

Lower score = more suspicious for all methods.

```bash
RUN=results/VerSe_classifier/label_override_8ep_23092026_2135
GLARE=<GLARE run dir>
OUT=<output dir>

# AUM forward sweep (~4.4 GPU-h, 19 266 samples × 8 checkpoints)
uv run python run_aum.py \
    --weights-dir $RUN/weights --spec-path configs/spec_lo_example.json --out-dir $OUT/aum

# VoG: input gradients (~12 GPU-h)
uv run python run_vog.py \
    --weights-dir $RUN/weights --spec-path configs/spec_lo_example.json --out-dir $OUT/vog

# Scores + metrics for GLARE, GLAREx, AUM, TracIn, TracIn-LL, VoG
uv run python evaluation/evaluate_run.py \
    --run-dir $GLARE --spec configs/spec_lo_example.json \
    --gt evaluation/gt_mislabels_derived.csv \
    --aum-dir $OUT/aum --vog-dir $OUT/vog --out-dir $OUT/evaluation
```

Smoke-test with `--limit-samples 20`.

---

## 5. Classifier evaluation (lower / upper bound, fraction checked)

Runs both classifiers on the test set and prints F1 / Accuracy / MCC.

```bash
# Standard test set (no anomaly correction):
uv run python evaluation/eval_lb_ub.py

# Anomaly-corrected test set (T13/T11/LabelOverride corrections applied):
uv run python evaluation/eval_lb_ub_anomaly_testset.py
```

The fraction-checked recall curves (recall at 5 % / 10 % / 20 % of training data inspected)
are part of the `evaluate_run.py` output written in step 4 (`$OUT/evaluation/evaluation_*.xlsx`).

---

## Project structure

| Path | Contents |
|---|---|
| `thesis_figures.ipynb` | Reproduces all 23 thesis figures from pre-computed data |
| `train_VerSe_densenet.py` | Classifier training (with LabelOverride) |
| `train_VerSe_upper_bound.py` | Classifier training (without LabelOverride, upper bound) |
| `glare.py` / `run_glare.py` | GLARE / GLAREx scoring |
| `run_aum.py` / `run_vog.py` | AUM and VoG baselines |
| `data/` | VerSe dataloaders, filter sheet, PID correction lists |
| `models/` | DenseNet-169 LightningModule |
| `helper/` | Checkpoint loading, last-layer TracIn, path resolution |
| `configs/` | Example training spec (`spec_lo_example.json`) |
| `evaluation/` | Ground-truth files, `evaluate_run.py`, evaluation scripts |
| `baselines_data/data/` | Pre-computed run outputs for all baseline figures (see manifest below) |
| `results/` | Classifier checkpoints (not in repo) |

---

## `baselines_data/data/` manifest

Pre-computed outputs copied 2026-10-01 from the thesis result folders (names unchanged):

| Path under `baselines_data/data/` | What | Read by |
|---|---|---|
| `glare/19-08-26_21:11_ep-8_single-gpu/results/{mislabels_list,grad_norms}_20260819_211102.*` | GLARE, no-LO 8-epoch run | notebook (no-LO figures) |
| `glare/26-09-26_07:42_ep-8_single-gpu/{mislabels_list,grad_norms}_20260926_074211.*`, `run_info.json` | GLARE, LO 8-epoch run | notebook, `evaluate_run.py --run-dir` |
| `runs/label_override_8ep_23092026_2135/spec.json`, `lr.json`, `epoch_metrics.json` | training spec + LR/metric log of the LO run | notebook |
| `glare/label_override_8ep_23092026_2135_baselines/evaluation/evaluation_20260928_232325.{xlsx,json}` | thesis baseline evaluation | notebook, Fig. 6.18 |
| `glare/label_override_8ep_23092026_2135_baselines/{confidence,vog}/` | raw AUM sweep (8 shards) and VoG (12 shards) | `evaluate_run.py --aum-dir/--vog-dir` |
| `glare/label_override_8ep_23092026_2135_baselines/spec_lo.json` | spec used for the baselines | `evaluate_run.py --spec` |

Not included: checkpoints (575 MB), CT crops (74 GB).
