# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 项目概述

本仓库是论文 *"An Accelerated Riemannian Conjugate Gradient Method and Its Application to Low-rank Matrix Completion"* 的 MATLAB 实现 + Python 数据预处理框架。核心是低秩矩阵补全的加速黎曼共轭梯度法，提供 7 种 CG 变体，并在合成数据、图像修复、块缺失、交通流、推荐系统 5 类场景下做对比实验。

## 运行方式

无构建系统、无测试框架、无 lint。`.m` 文件为 MATLAB 脚本/函数，`.py` 脚本用 `uv run python` 运行（不要直接 `python`，避免 Windows Store 存根）。

- **核心算法** `matrix_completion.m` 是函数：
  `[X, errors, metrics] = matrix_completion(A_observed, Omega, k, max_iter, tol, method)`
  - `method`：`'HZ' | 'DY' | 'FR' | 'PRP' | 'HS' | 'NHS' | 'Alg1'`（大小写敏感）。
  - `errors`：单列逐迭代梯度范数（向后兼容，老调用 `[X,errors] = ...` 不变）。
  - `metrics`：可选第 3 输出，`metrics.MAE` / `metrics.RMSE` 为观测集上的逐迭代误差，供 recommender 画收敛曲线。
- **路径自定位**：所有 MATLAB 脚本用 `fileparts(mfilename('fullpath'))` 推算仓库根，Python 脚本用 `os.path.dirname(os.path.abspath(__file__))`。**脚本可从任意目录运行**，无需 cd。
- **数据/产物不入库**：`data/`、`results/`、`figure/`、`text/` 均被 `.gitignore` 忽略。原始数据集需用户自行下载到 `data/`（见 README 的 Data Preparation）。
- **Python 依赖**：见 `requirements.txt`（`piq` 仅 `Verify_.py` 需要）。
- **MATLAB 工具箱**：Image Processing Toolbox（`psnr`/`ssim`/`imread` 等）、`svds`。

## 架构要点

### 单文件核心：`matrix_completion.m`

顶层函数 `matrix_completion` + 若干 nested local functions：
- `wolfe_line_search`（Armijo `ρ=1e-4` + 曲率 `σ=0.6`，二分区间，最多 20 步）
- `objective_function`（`0.5*‖P_Ω(X-A)‖_F²`）
- `compute_riemannian_gradient`（切空间投影）
- `retraction`（SVD 回撤，取前 `k` 奇异值）
- `vector_transport`（投影式向量传输）
- `compute_beta`（7 种 β 公式 + 范数缩放 + FR/DY/NHS/Alg1 的 `max(0,β)` 截断；HZ 有修正项）
- `P_Omega`（`M(~Omega)=0`）

迭代中 `X` 是 struct（`U,S,V`），函数末尾才合成 `X = U*S*V'`。`Alg1` 走独立 `case` 分支做二次回撤（`gamma_k = -a_k/b_k`），其余方法走 `otherwise` 单步回撤。`compute_beta` 里 `Alg1` 与 `NHS` 的 β 公式目前相同。

### 实验脚本模式

`experiments/*.m` 共享结构：参数 → 读取/生成 `A_observed`+`Omega` → 对 7 种 method 循环调 `matrix_completion` → `tic/toc`+指标 → 画图存 `results/`（synthetic 存 `figure/`、`text/`）。每个脚本在开头先 `mkdir` 所需输出目录。

### 数据流向

- 推荐系统：`movie_rating_processor.py`（`data/ratings.csv`）→ `data/rating_fillna.csv` + `data/rating_mask.csv` → `recommender_experiment.m` 读取。
- 交通：`PEM_data_process.py`（`data/data.npz`）→ `data/PE_data.csv` + `data/PE_mask.csv`（1=观测）→ `traffic_experiment.m` 读取。
- 图像：`create_lowrank_image(rank)` / `create_lowrank_color_image(rank)`（`data/test_image.png`）→ `results/rank{N}/rank_{N}_image.png` → `image_recovery_experiment.m`（rank=80）/ `block_mask_experiment.m`（rank=50）读取。
- 评估：`calculate_psnr.m`（函数）、`Verify_.py`（piq 独立核算 PSNR/SSIM）、`svd_analysis.py`（估秩）均为可选辅助工具。`rating_algorithm_comparison.py` 跑 MF/Bias-SVD/SVD++ 基线对照。

## 约定

- 注释/`fprintf` 输出中英文混用（实验脚本偏中文），新增代码沿用周边风格。
- `method` 字符串在主函数 `case`、`compute_beta` 的 `switch`、各实验 `methods` 元胞数组三处出现，改名需同步。
- 改 `matrix_completion.m` 时注意：`errors` 必须保持单列（其他实验按单列用），新指标一律走第 3 输出 `metrics`。