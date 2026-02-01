import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.decomposition import TruncatedSVD
import matplotlib.pyplot as plt
import os


DATA_DESCRIPTION = {
    "Flow": "单位时间内通过该检测器的车辆数量（通常每5分钟）",
    "Occupancy": "车辆占据检测器的时间比例，与密度相关",
    "Speed": "该检测器所在车道的车辆平均速度"
}


def load_pems_data(file_path):
    print("Loading PEMS data...")
    data_npz = np.load(file_path)
    data = data_npz['data']

    T, N, C = data.shape
    data = data.reshape(T, N)

    print(f"Original data shape: {data.shape}")
    return data


def prepare_data(data, max_rows=None):
    if max_rows is not None and max_rows > 0:
        data = data[:max_rows, :]
        print(f"Truncated data shape: {data.shape}")

    return data


def create_mask(data, missing_value=0):
    missing_mask = (data == missing_value)
    data_with_nan = data.copy()
    data_with_nan[missing_mask] = np.nan

    print(f"Missing values: {missing_mask.sum()}")
    return missing_mask, data_with_nan


def normalize_data(data):
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)
    return data_scaled, scaler


def save_processed_data(data, mask, output_dir="."):
    data_df = pd.DataFrame(data)
    mask_df = pd.DataFrame(mask)

    data_path = os.path.join(output_dir, "PE_data.csv")
    mask_path = os.path.join(output_dir, "PE_mask.csv")

    data_df.to_csv(data_path, index=False)
    mask_df.to_csv(mask_path, index=False)

    print(f"Data saved to: {data_path}")
    print(f"Mask saved to: {mask_path}")


def mean_absolute_percentage_error(y_true, y_pred):
    mask = y_pred != 0
    if np.sum(mask) == 0:
        return np.inf
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100


def matrix_completion_with_svd(X_nan, rank=20, max_iters=10, verbose=True):
    imputer = SimpleImputer(strategy='mean')
    X_filled = imputer.fit_transform(X_nan)

    mask = np.isnan(X_nan)

    for i in range(max_iters):
        if verbose:
            diff = (X_filled[mask] - X_nan[mask])
            mae = np.mean(np.abs(diff))
            print(f"Iteration {i+1} | Reconstruction MAE on missing: {mae:.4f}")

        svd = TruncatedSVD(n_components=rank)
        svd.fit(X_filled)
        U = svd.transform(X_filled)
        V = svd.components_

        X_filled = U @ V
        X_filled[~mask] = X_nan[~mask]

    return X_filled


def graph_regularized_matrix_factorization(data_with_nan, adj, max_rows, rank=20, alpha=0.1, max_iters=10):
    from sklearn.decomposition import NMF
    from scipy.sparse.linalg import eigsh

    D = np.diag(np.sum(adj, axis=1))
    L = D - adj

    W = np.random.rand(max_rows, rank)
    H = np.random.rand(rank, data_with_nan.shape[1])

    mask_mat = ~np.isnan(data_with_nan)

    for iter_num in range(max_iters):
        print(f"【GraphRegMF - Iteration {iter_num + 1}】")

        for t in range(max_rows):
            idx = mask_mat[t]
            if not np.any(idx):
                continue
            Wt = W[t:t+1, :]
            H_sub = H[:, idx]
            Xt = data_with_nan[t:t+1, idx]
            Wt_new = np.linalg.solve(H_sub @ H_sub.T + 1e-6 * np.eye(rank),
                                     H_sub @ Xt.T).T
            W[t] = Wt_new

        for n in range(data_with_nan.shape[1]):
            idx = mask_mat[:, n]
            if not np.any(idx):
                continue
            Hn = H[:, n:n+1]
            W_sub = W[idx]
            Xn = data_with_nan[idx, n:n+1]
            Hn_new = np.linalg.solve(W_sub.T @ W_sub + alpha * L[n:n+1, :].sum() * np.eye(rank) + 1e-6 * np.eye(rank),
                                     W_sub.T @ Xn)
            H[:, n:n+1] = Hn_new

    return W @ H


def time_aware_matrix_completion(data_with_nan, max_rows, rank=20, max_iters=10):
    time_weight = np.linspace(1, 0.5, max_rows).reshape(-1, 1)
    data_weighted = data_with_nan * time_weight

    filled_timeaware = matrix_completion_with_svd(data_weighted, rank=rank, max_iters=max_iters, verbose=False)
    return filled_timeaware


def evaluate_imputation(true_data, filled_data, mask):
    true_values = true_data[mask]
    imputed_values = filled_data[mask]

    mae = mean_absolute_error(true_values, imputed_values)
    rmse = np.sqrt(mean_squared_error(true_values, imputed_values))
    r2 = r2_score(true_values, imputed_values)
    mape = mean_absolute_percentage_error(true_values, imputed_values)

    return {
        "MAE": mae,
        "RMSE": rmse,
        "MAPE": mape,
        "R²": r2
    }


def visualize_imputation(true_data, filled_data, mask, sensor_id=0, save_path=None):
    plt.figure(figsize=(12, 4))
    plt.plot(true_data[:, sensor_id], label="True", alpha=0.7)
    plt.plot(filled_data[:, sensor_id], label="LowRankSVD", linestyle='--')

    missing_indices = np.where(mask[:, sensor_id])[0]
    imputed_values = filled_data[missing_indices, sensor_id]
    plt.scatter(missing_indices, imputed_values, color='red', s=10, label='Imputed Missing')

    plt.title(f"Sensor {sensor_id} Imputation Result (LowRankSVD)")
    plt.legend()
    plt.xlabel("Time Step")
    plt.ylabel("Flow")
    plt.grid(True)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {save_path}")
    else:
        plt.show()


def process_traffic_data(file_path, max_rows=10000, output_dir=".", plot_sensor=0):
    print("=" * 60)
    print("PEM Traffic Data Processing")
    print("PEM交通流数据处理")
    print("=" * 60)

    print("\nData Description:")
    for key, value in DATA_DESCRIPTION.items():
        print(f"{key}: {value}")

    print("\n1. Loading data...")
    data = load_pems_data(file_path)

    print("\n2. Preparing data...")
    data = prepare_data(data, max_rows)

    print("\n3. Creating mask...")
    missing_mask, data_with_nan = create_mask(data)

    print("\n4. Saving processed data...")
    save_processed_data(data, missing_mask, output_dir)

    print("\n5. Normalizing data...")
    data_scaled, scaler = normalize_data(data)

    print("\n6. Running imputation methods...")

    print("\n--- Method 1: Low-rank SVD completion ---")
    filled_softimpute = matrix_completion_with_svd(data_with_nan, rank=20, max_iters=10)

    print("\n--- Method 2: KNN Imputation ---")
    knn_imputer = KNNImputer(n_neighbors=5)
    filled_knn = knn_imputer.fit_transform(data_with_nan)

    print("\n--- Method 3: Mean Imputation ---")
    mean_imputer = SimpleImputer(strategy='mean')
    filled_mean = mean_imputer.fit_transform(data_with_nan)

    print("\n--- Method 4: Median Imputation ---")
    median_imputer = SimpleImputer(strategy='median')
    filled_median = median_imputer.fit_transform(data_with_nan)

    print("\n--- Method 5: Graph Regularized Matrix Factorization ---")
    try:
        adj = np.load(file_path)['adj']
        filled_grmf = graph_regularized_matrix_factorization(
            data_with_nan, adj, max_rows, rank=20, max_iters=10
        )
    except KeyError:
        print("Adjacency matrix not found, skipping Graph Regularized MF")
        filled_grmf = None

    print("\n--- Method 6: Time-Aware Matrix Factorization ---")
    filled_timeaware = time_aware_matrix_completion(data_with_nan, max_rows, rank=20, max_iters=10)

    print("\n7. Evaluating all methods...")
    methods = {
        "LowRankSVD": filled_softimpute,
        "KNNImputer": filled_knn,
        "MeanImputer": filled_mean,
        "MedianImputer": filled_median,
        "GraphRegMF": filled_grmf,
        "TimeAwareMF": filled_timeaware,
    }

    results = {}
    for name, filled_data in methods.items():
        if filled_data is not None:
            results[name] = evaluate_imputation(data, filled_data, missing_mask)

    print("\n" + "="*40)
    print("Imputation Results:")
    print("="*40)
    for method_name, metrics in results.items():
        print(f"\n【{method_name}】")
        print(f"MAE: {metrics['MAE']:.4f}")
        print(f"RMSE: {metrics['RMSE']:.4f}")
        print(f"MAPE: {metrics['MAPE']:.2f}%")
        print(f"R²: {metrics['R²']:.4f}")

    print("\n8. Visualizing results...")
    plot_path = os.path.join(output_dir, "imputation_visualization.png") if output_dir else None
    visualize_imputation(data, filled_softimpute, missing_mask, sensor_id=plot_sensor, save_path=plot_path)

    print("\n" + "="*60)
    print("Processing completed successfully!")
    print("处理完成！")
    print("="*60)


if __name__ == "__main__":
    file_path = "data.npz"
    process_traffic_data(file_path, max_rows=10000, output_dir=".")