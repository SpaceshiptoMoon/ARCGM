# A Safeguarded Accelerated Riemannian Conjugate-Gradient Method for Low-Rank Matrix Completion

**Ya-qiong Wen¹, Lin-hui Liu¹, Jiao-fen Li¹·²**

1. *School of Mathematics and Computing Science, Guangxi Colleges and Universities Key Laboratory
   of Data Analysis and Computation, Guilin University of Electronic Technology, Guilin 541004, China*
2. *Center for Applied Mathematics of Guangxi (GUET), Guilin 541004, China*

---

## Abstract

We consider least-squares matrix completion on the smooth manifold of matrices with prescribed
rank. A Riemannian conjugate-gradient scheme is developed by combining an improved
Hestenes–Stiefel-type coefficient with a secant-based scaling of the accepted line-search step.
The scaling is **safeguarded**: an accelerated trial point is used only when it satisfies the same
strong Wolfe conditions as the baseline step, and the conjugate direction is restarted whenever it
fails uniform descent or norm bounds. These safeguards resolve the mismatch that otherwise arises
between an accelerated update and a convergence proof based on the unscaled line-search point.
Under the assumptions of compact level sets and a uniformly Lipschitz continuous pullback
derivative, the full gradient sequence is guaranteed to converge, as established via the
Riemannian–Zoutendijk condition. Numerical experiments on synthetic instances, MovieLens ratings,
PeMS traffic data, and image inpainting illustrate the practical behavior of the method relative
to several classical Riemannian conjugate-gradient formulas and application-specific baselines.

**Mathematics Subject Classification:** 65K10; 90C30; 15A83

**Keywords:** Low-rank matrix completion; Riemannian optimization; conjugate-gradient method;
fixed-rank manifold; strong Wolfe line search

---

## Repository overview

This repository contains the MATLAB implementation and the experimental framework for the paper
*"A Safeguarded Accelerated Riemannian Conjugate-Gradient Method for Low-Rank Matrix Completion"*.

It implements the safeguarded accelerated Riemannian conjugate-gradient (ARCG) method with a
strong Wolfe line search for **low-rank matrix completion** — recovering a low-rank matrix `X`
from a small set of observed entries `P_Ω(X)` of a matrix `A`. The proposed method combines an
improved Hestenes–Stiefel-type (NHS) coefficient with a secant-based scaling of the accepted
line-search step, safeguarded by strong Wolfe acceptance and restart on failure of uniform
descent / norm bounds. Seven CG update variants share one unified interface and are compared
across five application scenarios: synthetic low-rank recovery, grayscale image inpainting,
block-missing image recovery, traffic-flow completion, and movie-rating prediction.

---

## Table of Contents

- [How the code is organized](#how-the-code-is-organized)
- [The shared experiment pipeline](#the-shared-experiment-pipeline)
- [Methods at a glance](#methods-at-a-glance)
- [Requirements](#requirements)
- [Data preparation](#data-preparation)
- [Reproducing each experiment](#reproducing-each-experiment)
- [Optional tooling](#optional-tooling)
- [Metrics](#metrics)
- [Citation](#citation)
- [License](#license)

---

## How the code is organized

```
ARCGM_Code_Part/
├── matrix_completion.m                 # Core ARCG algorithm: 7 CG variants + Wolfe line search
│
├── experiments/                        # One script per application scenario
│   ├── synthetic_experiment.m          #   1. Synthetic low-rank recovery
│   ├── image_recovery_experiment.m     #   2. Grayscale image inpainting (random mask)
│   ├── block_mask_experiment.m         #   3. Block-missing image recovery
│   ├── traffic_experiment.m            #   4. Traffic-flow completion (PEMS)
│   └── recommender_experiment.m        #   5. Movie-rating prediction (MovieLens)
│
├── data_process/                       # Data preprocessing & evaluation helpers
│   ├── image/
│   │   ├── create_lowrank_image.m      #   Build a low-rank grayscale image (SVD truncation)
│   │   └── create_lowrank_color_image.m#   Build a low-rank color image
│   ├── r_estimate/
│   │   ├── calculate_psnr.m            #   PSNR between two images (MATLAB)
│   │   ├── svd_analysis.py             #   SVD analysis to help choose the rank k
│   │   └── Verify_.py                  #   Independent PSNR/SSIM check via piq
│   ├── recommendation_system/
│   │   ├── movie_rating_processor.py   #   MovieLens -> rating matrix + observation mask
│   │   └── rating_algorithm_comparison.py  # MF / Bias-SVD / SVD++ baselines
│   └── traffic/
│       └── PEM_data_process.py         #   PEMS .npz -> flow matrix + observation mask
│
├── data/                               # (gitignored) raw + preprocessed datasets land here
├── results/                            # (gitignored) experiment outputs
├── figure/  text/                      # (gitignored) synthetic-experiment outputs
├── requirements.txt
├── LICENSE
└── README.md
```

**One important convention:** every MATLAB and Python script locates the repository root from its
own file path (`mfilename('fullpath')` in MATLAB, `__file__` in Python). You can therefore run any
script **from any working directory** — no `cd` into a specific folder is required.

---

## The shared experiment pipeline

All five experiments follow the same three-stage pattern. Understanding this pattern makes the
whole repository predictable:

```
  [1] raw data            [2] preprocessed input          [3] experiment
  (download / own)   -->   (A_observed + Omega)      -->   matrix_completion  -->  figures + tables
        |                       ^                              ^                      ^
   place in data/         scripts in data_process/        scripts in experiments/    results/
```

1. **Raw data** — you provide the original dataset (downloaded from standard sources, or your own
   images) into `data/`.
2. **Preprocessed input** — a `data_process/` script (or `create_lowrank_image`) turns the raw
   data into two things every experiment needs:
   - `A_observed`: the observed matrix, with **0 at unobserved positions**;
   - `Omega`: a logical **observation mask**, **1 = observed, 0 = missing**.
3. **Experiment** — an `experiments/*.m` script loads `A_observed` and `Omega`, runs
   `matrix_completion` once per CG variant, collects timing + convergence + metrics, and writes
   figures and tables to `results/`.

The core call, identical across all experiments, is:

```matlab
[X, errors, metrics] = matrix_completion(A_observed, Omega, k, max_iter, tol, method);
```

- `X`        — recovered matrix
- `errors`   — per-iteration Riemannian gradient norm (convergence curve)
- `metrics`  — optional: per-iteration MAE / RMSE on the observed set (used by the recommender
  experiment to plot error-convergence curves)

So to reproduce any experiment you only need to know: **which raw data** → **which preprocessor**
→ **which experiment script**. The next sections give exactly that, per experiment.

---

## Methods at a glance

The unified CG update is `η_{k+1} = -∇f(x_{k+1}) + β_k η_k`, with `β_k` chosen by the `method`
argument:

| `method` | β formula family |
|----------|------------------|
| `HZ`     | Hager–Zhang (correction term, non-negative) |
| `DY`     | Dai–Yuan |
| `FR`     | Fletcher–Reeves |
| `PRP`    | Polak–Ribière–Polyak |
| `HS`     | Hestenes–Stiefel |
| `NHS`    | improved Hestenes–Stiefel-type (hybrid, non-negative) — the coefficient used by the proposed method |
| `Alg1`   | **the proposed safeguarded accelerated variant**: NHS coefficient + secant-based scaling of the accepted step, with safeguarded acceptance (strong Wolfe) and restart on failure of uniform descent / norm bounds |

**Wolfe line search** uses the strong Wolfe conditions (Armijo `ρ = 1e-4`, curvature `σ = 0.6`).
Manifold operations: SVD-based **retraction**, projection-based **vector transport**, tangent-space
**gradient projection**, and the observation projection `P_Ω`.

---

## Requirements

- **MATLAB** R2018b+ with Image Processing Toolbox (`imread`, `im2double`, `rgb2gray`, `psnr`,
  `ssim`, `imshow`) and `svds` support.
- **Python** 3.12+ — only for data preprocessing and the optional PSNR/SSIM verification:

  ```bash
  uv pip install -r requirements.txt     # or: pip install -r requirements.txt
  ```

---

## Data preparation

Datasets are **not** bundled (size/licensing). Only the **standard / official sources** of the
raw data are listed below; all derived matrices are generated automatically by the preprocessing
scripts. Place raw files under `data/` (gitignored).

| Experiment | Raw data | Standard source | Save as |
|---|---|---|---|
| Recommender | MovieLens ratings | <https://grouplens.org/datasets/movielens/> | `data/ratings.csv` |
| Traffic | PEMS traffic | Caltrans PeMS portal <http://pems.dot.ca.gov/> | `data/data.npz` |
| Image (×2) | your own images | — (no standard source) | `data/test_image.png`, `data/test_color_image.png` |
| Synthetic | none (generated in-code) | — | — |

The preprocessors that turn these into `A_observed` + `Omega` are invoked in each experiment's
section below.

> **PEMS format note:** `PEM_data_process.py` expects the common research-packaged `.npz` with a
> `data` key of shape `T × N × C` (and optionally an `adj` adjacency key). If you start from raw
> PeMS portal CSVs, package them into this `.npz` layout first.

---

## Reproducing each experiment

Run all MATLAB scripts from any directory. Each iterates over all seven methods, records timing
and convergence, and writes figures + tables under `results/` (or `figure/`, `text/` for the
synthetic case).

### 1. Synthetic low-rank recovery — `synthetic_experiment.m`

**What it tests:** pure algorithm behavior on a problem with known ground truth. A random
rank-`k` matrix `A_true = U_true V_trueᵀ` is generated, a fraction of entries is observed
(controlled by the oversampling factor `OS`), and each CG variant tries to recover `A_true`.

**Data:** none — generated inside the script.

**Run:**
```matlab
synthetic_experiment
```
**Key parameters (top of script):** `m = 200, n = 200, k_true = 10, OS = 5, max_iter = 200, tol = 1e-8`.
`OS` is the oversampling factor: `num_observations = OS · (m + n − k) · k`.

**Outputs:** `figure/n=200_m=200_r=10_os=5.eps` (gradient-norm convergence, log y) and
`text/n=200_m=200_r=10_os=5.txt` (per-method iterations, time, reconstruction error, final
gradient norm). Reconstruction error `‖X − A_true‖_F / ‖A_true‖_F` is reported because `A_true` is
known.

### 2. Image inpainting (random mask) — `image_recovery_experiment.m`

**What it tests:** recovering a low-rank grayscale image after a random fraction of pixels is
removed. Quality is measured by PSNR / SSIM against the original.

**Pipeline:**
```matlab
% (a) put your image at data/test_image.png
create_lowrank_image(80)            % builds the low-rank input results/rank80/rank_80_image.png
image_recovery_experiment           % runs all 7 methods
```
**Key parameters:** `rank = 80, os = 0.7, max_iter = 100, tol = 1e-12`. Here `os` is the
**missing ratio** — a pixel is dropped with probability `os`, so `1 − os` of pixels are observed
(note this is the opposite sense from the synthetic `OS`).

**Outputs (in `results/rank80/os0.7/`):** per-method `<Method>.eps` / `.png` recovered images,
`damaged_image.*`, `convergence_comparison_log.*`, `summary_results.*` (3×3 grid), and
`performance_results.txt` (PSNR / SSIM / time per method).

### 3. Block-missing recovery — `block_mask_experiment.m`

**What it tests:** recovery when a contiguous `block × block` region (centered) is missing, rather
than scattered pixels. This is harder for matrix completion and stresses the method's
generalization.

**Pipeline:**
```matlab
create_lowrank_image(50)            % builds results/rank50/rank_50_image.png
block_mask_experiment               % currently demonstrates the HZ method
```
**Key parameters:** `rank = 50, block_size = 100, max_iter = 90, tol = 1e-12`. The `methods` cell
at the top defaults to `{'HZ'}`; expand it to all seven for a full comparison.

**Outputs (in `results/rank50/block100/`):** `HZ.eps/.png`, `damaged_image.*`,
`convergence_comparison_log.*`, `summary_results.*`, `performance_results.txt`.

### 4. Traffic-flow completion — `traffic_experiment.m`

**What it tests:** completing a traffic-flow matrix (time steps × sensors) where some readings are
missing, using real PEMS data.

**Pipeline:**
```bash
# (a) put PEMS data at data/data.npz
cd data_process/traffic
uv run python PEM_data_process.py     # -> data/PE_data.csv, data/PE_mask.csv (1 = observed)
```
```matlab
traffic_experiment                   % runs all 7 methods
```
**Key parameters:** `k_true = 40, max_iter = 1000, tol = 1e-8`. The matrix size `m × n` is read
automatically from `PE_data.csv`. Reported metrics are MAE / RMSE on the observed entries plus the
gradient-norm convergence curve.

**Outputs:** `results/table/n=..._r=40.eps` (convergence) and `results/text/n=..._r=40.txt`
(per-method iterations, time, MAE, final gradient norm).

### 5. Movie-rating prediction — `recommender_experiment.m`

**What it tests:** completing a sparse user–movie rating matrix (MovieLens), predicting missing
ratings. Convergence is tracked by MAE / RMSE on the observed entries at every iteration.

**Pipeline:**
```bash
# (a) put MovieLens ratings at data/ratings.csv
cd data_process/recommendation_system
uv run python movie_rating_processor.py   # -> data/rating_fillna.csv, data/rating_mask.csv
```
```matlab
recommender_experiment               % runs all 7 methods
```
**Key parameters:** `m = 610` users, `n = 1000` movies, `k_true = 10, max_iter = 50, tol = 1e-8`.
The per-iteration MAE / RMSE histories come from the `metrics` output of `matrix_completion`.

**Outputs:** `results/table/MAE_convergence_lines_only.eps`,
`results/table/RMSE_convergence_lines_only.eps`, and
`results/text/n=1000_m=610_r=10_iterations.txt` (final MAE / RMSE / time per method).

---

## Optional tooling

These are not required to reproduce the main results, but support the workflow:

- **Independent PSNR/SSIM check** — `data_process/r_estimate/Verify_.py` recomputes image metrics
  with the `piq` library to cross-check the MATLAB side:

  ```bash
  # single pair
  uv run python data_process/r_estimate/Verify_.py \
      --recovered results/rank80/os0.7/HZ.png \
      --original results/rank80/rank_80_image.png
  # batch over all methods in a directory
  uv run python data_process/r_estimate/Verify_.py \
      --dir results/rank80/os0.7 --origin results/rank80/rank_80_image.png
  ```

- **Rank estimation** — `data_process/r_estimate/svd_analysis.py` plots singular-value / cumulative
  variance / approximation-error curves to help choose the initial rank `k`:

  ```bash
  uv run python data_process/r_estimate/svd_analysis.py \
      --image results/rank80/original_grayscale.png --max-k 100
  ```

- **Baseline recommender algorithms** — `data_process/recommendation_system/rating_algorithm_comparison.py`
  compares MF, Bias-SVD and SVD++ baselines on the same rating matrix (run after
  `movie_rating_processor.py`):

  ```bash
  uv run python data_process/recommendation_system/rating_algorithm_comparison.py
  ```

- **Quick PSNR** — `calculate_psnr(path1, path2)` in MATLAB returns the PSNR between two image
  files.

---

## Metrics

- **Gradient norm** `‖grad‖_F` — the convergence criterion, plotted for every experiment.
- **Reconstruction error** `‖X − X_true‖_F / ‖X_true‖_F` — synthetic only (ground truth known).
- **PSNR / SSIM** — image experiments (MATLAB `psnr` / `ssim`, cross-checked with `piq`).
- **MAE / RMSE** — recommender (per-iteration via `metrics`) and traffic (on observed entries).

---

## Citation

If you use this code, please cite the accompanying paper:

```bibtex
@article{ARCGM,
  title   = {A Safeguarded Accelerated Riemannian Conjugate-Gradient Method
             for Low-Rank Matrix Completion},
  author  = {Wen, Ya-qiong and Liu, Lin-hui and Li, Jiao-fen},
  journal = {},
  year    = {2026},
  note    = {Mathematics Subject Classification: 65K10; 90C30; 15A83}
}
```

---

## License

Released under the [MIT License](LICENSE).