# An Accelerated Riemannian Conjugate Gradient Method and Its Application to Low-rank Matrix Completion

This repository contains the MATLAB implementation for the paper *"An Accelerated Riemannian Conjugate Gradient Method and Its Application to Low-rank Matrix Completion"*. The implementation includes comprehensive algorithms and experimental frameworks for matrix completion problems.

## Repository Structure

### Core Algorithm
- **[`matrix_completion.m`](matrix_completion.m)** - Main implementation of the accelerated Riemannian conjugate gradient method with Wolfe line search, supporting 7 different CG variants (HZ, DY, FR, PRP, HS, NHS, Alg1)

### Experiments
The [`experiments/`](experiments/) folder contains scripts for testing the algorithm on diverse applications:

- **[`synthetic_experiment.m`](experiments/synthetic_experiment.m)** - Synthetic matrix completion experiments comparing all 7 CG methods on randomly generated low-rank matrices
- **[`image_recovery_experiment.m`](experiments/image_recovery_experiment.m)** - Image inpainting experiments using grayscale images with random masking, evaluating PSNR and SSIM metrics
- **[`block_mask_experiment.m`](experiments/block_mask_experiment.m)** - Block missing data recovery experiments using 100×100 pixel block masks at the image center
- **[`traffic_experiment.m`](experiments/traffic_experiment.m)** - Traffic data completion experiments using PEMS dataset for traffic flow prediction
- **[`recommender_experiment.m`](experiments/recommender_experiment.m)** - Recommender system experiments using MovieLens dataset, tracking MAE and RMSE metrics

### Data Processing
The [`data_process/`](data_process/) folder contains utilities for data preparation and evaluation:

- **[`image/`](data_process/image/)** - Image processing utilities:
  - [`create_lowrank_image.m`](data_process/image/create_lowrank_image.m) - Creates low-rank grayscale images via SVD decomposition for testing
  - [`create_lowrank_color_image.m`](data_process/image/create_lowrank_color_image.m) - Creates low-rank color images via channel-wise SVD decomposition
- **[`r_estimate/`](data_process/r_estimate/)** - Evaluation utilities:
  - [`calculate_psnr.m`](data_process/r_estimate/calculate_psnr.m) - Calculates Peak Signal-to-Noise Ratio between two images
- **[`recommendation_system/`](data_process/recommendation_system/)** - Python utilities for recommendation system data:
  - [`movie_rating_processor.py`](data_process/recommendation_system/movie_rating_processor.py) - Processes movie rating data and creates rating matrices
  - [`rating_algorithm_comparison.py`](data_process/recommendation_system/rating_algorithm_comparison.py) - Compares different rating algorithms (MF, Bias-SVD, SVD++)

## Methods

### Wolfe Line Search

The implementation employs a sophisticated line search algorithm satisfying both:

- **Armijo condition**: Ensures sufficient decrease in the objective function
- **Curvature condition**: Ensures adequate gradient orthogonality

**Parameters**:
- `rho`: Armijo condition parameter (default: 1e-4)
- `sigma`: Curvature condition parameter (default: 0.6)

### Conjugate Gradient Variants

Each method implements the unified update rule:
```
η_{k+1} = -∇f(x_{k+1}) + β_k η_k
```

where β_k is computed using distinct formulas for each variant.

### Manifold Operations

- **Retraction**: Maps tangent vectors back to the manifold using SVD
- **Vector Transport**: Transports vectors between tangent spaces
- **Projection**: Projects gradients onto the tangent space

## Performance Metrics

The implementation tracks comprehensive performance metrics:

- **Gradient Norm**: Convergence measure
- **Reconstruction Error**: $\frac{\|X - X_{\text{true}}\|_F}{\|X_{\text{true}}\|_F}$
- **PSNR**: Peak Signal-to-Noise Ratio (for images)
- **SSIM**: Structural Similarity Index (for images)
- **MAE/RMSE**: Mean Absolute Error/Root Mean Square Error (for ratings)

## Directory Structure

```
ARCGM_Code_Part/
├── matrix_completion.m                    % Main algorithm
├── experiments/
│   ├── synthetic_experiment.m            % Synthetic data experiments
│   ├── image_recovery_experiment.m        % Image recovery experiments
│   ├── block_mask_experiment.m            % Block missing data
│   ├── traffic_experiment.m               % Traffic data experiments
│   └── recommender_experiment.m           % Recommender system experiments
└── data_process/
    ├── image/                            % Image processing utilities
    ├── r_estimate/                       % PSNR calculation utilities
    └── recommendation_system/             % Recommendation system utilities
```