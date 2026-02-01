import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from PIL import Image
import os

plt.rcParams["font.sans-serif"] = ["Arial Unicode MS", "DejaVu Sans", "Liberation Sans"]
plt.rcParams["axes.unicode_minus"] = False  # 解决负号显示问题

def svd_analysis(matrix, matrix_name="Observation Matrix", variance_threshold1=0.97, variance_threshold2=0.99, max_k=20, save_directory=".", save_format="png"):
    """
    对矩阵进行完整的SVD分析，帮助选择矩阵填充的初始秩
    
    参数:
    matrix: 待分析的矩阵
    matrix_name: 矩阵名称（用于图表标题）
    variance_threshold1: 第一个方差贡献率阈值（默认97%）
    variance_threshold2: 第二个方差贡献率阈值（默认99%）
    max_k: 近似误差分析中最大k值（默认20）
    save_directory: 保存目录
    save_format: 保存格式，可选 'png' 或 'eps'，默认为 'png'
    """
    print(f"=== {matrix_name} SVD Analysis Report ===")
    print(f"Matrix Shape: {matrix.shape}")
    print(f"Matrix Sparsity: {np.sum(matrix == 0) / matrix.size:.2%}")

    if not os.path.exists(save_directory):
        os.mkdir(save_directory)
    
    # 1. 数据标准化（可选，对于量纲不一致的数据很重要）
    scaler = StandardScaler(with_std=False)
    matrix_centered = scaler.fit_transform(matrix)
    
    # 2. 进行SVD分解
    U, sigma, Vt = np.linalg.svd(matrix_centered, full_matrices=False)
    
    print(f"\nNumber of Singular Values: {len(sigma)}")
    print(f"First 10 Singular Values: {sigma[:10]}")
    
    # 3. 计算累积方差贡献率
    explained_variance = (sigma ** 2) / np.sum(sigma ** 2)
    cumulative_variance = np.cumsum(explained_variance)
    
    # 4. 找到达到不同阈值所需的最小秩
    k_threshold_1 = np.argmax(cumulative_variance >= variance_threshold1) + 1
    k_threshold_2 = np.argmax(cumulative_variance >= variance_threshold2) + 1
    

    print(f"Rank required to reach {variance_threshold1:.0%} variance contribution: {k_threshold_1}")
    print(f"Rank required to reach {variance_threshold2:.0%} variance contribution: {k_threshold_2}")
    print(f"Cumulative contribution of first {k_threshold_2} singular values: {cumulative_variance[k_threshold_2-1]:.3%}")
    
    # 5. 可视化分析 - 分别创建三个子图并保存
    fig1 = plt.figure(figsize=(10, 6))
    
    # 子图1: 奇异值分布
    plt.plot(range(1, len(sigma) + 1), sigma, 'bo-', linewidth=2, markersize=4)
    
    # 添加多条垂直虚线表示不同阈值对应的秩
    plt.axvline(x=k_threshold_1, color='green', linestyle='--', linewidth=1.5, 
                alpha=0.7, label=f'{variance_threshold1:.0%} Threshold (k={k_threshold_1})')
    plt.axvline(x=k_threshold_2, color='red', linestyle='-.', linewidth=2, 
                alpha=0.8, label=f'{variance_threshold2:.0%} Threshold (k={k_threshold_2})')
    
    plt.xlabel('Singular Value Index')
    plt.ylabel('Singular Value Magnitude')
    plt.title(f'{matrix_name} - Singular Value Distribution\n(Finding the elbow point)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # 保存第一个图表
    plt.savefig(os.path.join(save_directory, f'{matrix_name}_Singular_Value_Distribution.{save_format}'), dpi=300, bbox_inches='tight')
    plt.close(fig1)
    
    # 第二个图表：累积方差贡献率
    fig2 = plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(cumulative_variance) + 1), cumulative_variance, 'go-', 
             linewidth=2, markersize=4)
    
    # 添加水平虚线表示不同阈值
    plt.axhline(y=variance_threshold1, color='green', linestyle='--', linewidth=1.5, 
                alpha=0.7, label=f'{variance_threshold1:.0%} Threshold')
    plt.axhline(y=variance_threshold2, color='red', linestyle='-.', linewidth=2, 
                alpha=0.8, label=f'{variance_threshold2:.0%} Threshold')
    
    # 添加垂直虚线标记对应的秩位置
    plt.axvline(x=k_threshold_1, color='green', linestyle='--', linewidth=1.5, alpha=0.3)
    plt.axvline(x=k_threshold_2, color='red', linestyle='-.', linewidth=1.5, alpha=0.3)
    
    plt.xlabel('Number of Retained Singular Values (k)')
    plt.ylabel('Cumulative Variance Contribution')
    plt.title(f'{matrix_name} - Cumulative Variance Contribution')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # 保存第二个图表
    plt.savefig(os.path.join(save_directory,f'{matrix_name}_Cumulative_Variance_Contribution.{save_format}'), dpi=300, bbox_inches='tight')
    plt.close(fig2)
    
    # 第三个图表：前k个秩的近似误差
    fig3 = plt.figure(figsize=(10, 6))
    approximation_errors = []
    
    # 使用传入的max_k参数，但不超过实际奇异值数量
    k_values = range(1, min(max_k, len(sigma)) + 1)
    
    for k in k_values:
        # 使用前k个奇异值重构矩阵
        U_k = U[:, :k]
        sigma_k = sigma[:k]
        Vt_k = Vt[:k, :]
        matrix_approx = U_k @ np.diag(sigma_k) @ Vt_k
        
        # 计算相对Frobenius范数误差
        error = np.linalg.norm(matrix_centered - matrix_approx, 'fro') / np.linalg.norm(matrix_centered, 'fro')
        approximation_errors.append(error)
    
    plt.plot(k_values, approximation_errors, 'mo-', linewidth=2, markersize=4)
    
    # 在近似误差图中标记关键点

    if k_threshold_1 <= max(k_values):
        plt.axvline(x=k_threshold_1, color='green', linestyle='--', linewidth=1.5, alpha=0.3)
        plt.plot(k_threshold_1, approximation_errors[k_threshold_1-1], 's', 
                 color='green', markersize=6, label=f'{variance_threshold1:.0%} Threshold Point')
    
    if k_threshold_2 <= max(k_values):
        plt.axvline(x=k_threshold_2, color='red', linestyle='-.', linewidth=1.5, alpha=0.3)
        plt.plot(k_threshold_2, approximation_errors[k_threshold_2-1], '^', 
                 color='red', markersize=6, label=f'{variance_threshold2:.0%} Threshold Point')
    
    plt.xlabel('Number of Retained Singular Values (k)')
    plt.ylabel('Relative Approximation Error (Frobenius Norm)')
    plt.title(f'{matrix_name} - Approximation Error Analysis')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # 保存第三个图表
    plt.savefig(os.path.join(save_directory,f'{matrix_name}_Approximation_Error_Analysis.{save_format}'), dpi=300, bbox_inches='tight')
    plt.close(fig3)
    
    print(f"\nCharts saved as:")
    print(f"- {matrix_name}_Singular_Value_Distribution.{save_format}")
    print(f"- {matrix_name}_Cumulative_Variance_Contribution.{save_format}") 
    print(f"- {matrix_name}_Approximation_Error_Analysis.{save_format}")
    
    return k_threshold_2, sigma, cumulative_variance


if __name__ == "__main__":
    # 示例: 分析灰度图像矩阵
    print("=== Image Matrix SVD Analysis ===")

    # 读取图像并转换为灰度矩阵
    image_path = r"C:\Users\xl\Desktop\论文二\results\rank60\original_grayscale.png"

    image = Image.open(image_path)
    image_array = np.array(image)
    print(f"Successfully read image, shape: {image_array.shape}")
    
    # 进行SVD分析，使用0.97和0.99阈值，最大k值设为50
    k_recommended, singular_values, cumulative_var = svd_analysis(
        image_array, 
        matrix_name="Image Grayscale Matrix", 
        variance_threshold1=0.97,
        variance_threshold2=0.99,
        max_k=100,  # 可以调整这个参数来控制近似误差分析的范围
        save_directory=r"C:\Users\xl\Desktop\论文二\coding\r_estimate_py\output",
        save_format="eps"  # 默认保存为PNG格式，可改为"eps"以保存为EPS格式
    )
    
    print(f"\nRecommended rank {k_recommended} for matrix completion or compression")
    print(f"Cumulative explained variance of first {k_recommended} singular values: {cumulative_var[k_recommended-1]:.3%}")