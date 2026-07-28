from PIL import Image
import numpy as np
import torch
import os
from pathlib import Path


def calculate_psnr(img1, img2):
    # 将PIL图像转换为numpy数组
    img1_array = img1
    img2_array = img2
    
    if img1_array.shape != img2_array.shape:
        raise ValueError("Input images must have the same size.")

    mse = np.mean((np.float64(img1_array) - np.float64(img2_array)) ** 2)
    if mse == 0:
        return float("inf")
    
    # 根据图像数据类型确定最大像素值
    max_pixel_value = 255.0 if img1_array.max() <= 255 else 1.0
    return 20 * np.log10(max_pixel_value / np.sqrt(mse))


def calculate_ssim_piq(img1, img2):
    """
    使用 piq 库计算 SSIM

    参数:
    img1: PIL Image对象或numpy数组，第一幅图像
    img2: PIL Image对象或numpy数组，第二幅图像

    返回:
    ssim_value: SSIM值，范围[0,1]
    """
    # 将PIL图像转换为numpy数组
    img1_array = img1
    img2_array = img2
    
    if img1_array.shape != img2_array.shape:
        raise ValueError("Input images must have the same size.")

    # 转换图像为Tensor，确保数据范围在[0,1]
    img1_array = img1_array.astype(np.float32) / 255.0
    img2_array = img2_array.astype(np.float32) / 255.0
    
    # 转换为PyTorch张量，形状为[batch, channels, height, width]
    if len(img1_array.shape) == 2:  # 灰度图像
        img1_tensor = torch.from_numpy(img1_array).unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
        img2_tensor = torch.from_numpy(img2_array).unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
    else:  # 彩色图像
        img1_tensor = torch.from_numpy(img1_array.transpose(2, 0, 1)).unsqueeze(0)  # [1, C, H, W]
        img2_tensor = torch.from_numpy(img2_array.transpose(2, 0, 1)).unsqueeze(0)  # [1, C, H, W]

    # 使用piq库计算SSIM
    import piq

    ssim_value = piq.ssim(img1_tensor, img2_tensor, data_range=1.0)

    return ssim_value.item()

def get_directory_list(dir_path: str)-> list:
    dir_list: list = []
    for file in os.listdir(dir_path):
        if file.endswith(".png"):
            dir_list.append(os.path.join(dir_path, file))
    return dir_list
    

def writer_txt(dir_path, image_path):
    result_path = os.path.join(dir_path, "result.txt")
    dir_list = get_directory_list(dir_path)
    image_origin = (np.array(Image.open(image_path)) / 65535.0 * 255).astype(np.uint8)

    with open(result_path, "w") as f:
        f.write("方法:\tPSNR\tSSIM\n")

    method = ['HZ', 'DY', 'FR', 'PRP', 'HS', 'NHS', 'Alg1']
    for path in dir_list:
        if Path(path).stem in method:
            image = np.array(Image.open(path))
            psnr = calculate_psnr(image_origin, image)
            ssim= calculate_ssim_piq(image_origin, image)


            with open(result_path, "a") as f:
                f.write(f"{Path(path).stem}\t{round(psnr, 2)}\t{round(ssim, 2)}\n")



if __name__ == "__main__":
    import argparse

    _THIS_DIR = os.path.dirname(os.path.abspath(__file__))
    _REPO_ROOT = os.path.dirname(os.path.dirname(_THIS_DIR))
    _RESULTS_DIR = os.path.join(_REPO_ROOT, "results")

    parser = argparse.ArgumentParser(description="用 piq 独立核算图像 PSNR/SSIM，与 MATLAB 侧对照")
    parser.add_argument("--recovered", help="单图模式：恢复图路径")
    parser.add_argument("--original", help="单图模式：原图路径（16-bit PNG 按 65535 归一再转 8 位）")
    parser.add_argument("--dir", help="批量模式：存放各方法恢复图 (HZ.png 等) 的目录")
    parser.add_argument("--origin", help="批量模式：原图路径")
    args = parser.parse_args()

    if args.dir and args.origin:
        writer_txt(args.dir, args.origin)
        print(f"结果已写入 {os.path.join(args.dir, 'result.txt')}")
    elif args.recovered and args.original:
        image_origin = (np.array(Image.open(args.original)) / 65535.0 * 255).astype(np.uint8)
        image = np.array(Image.open(args.recovered))
        psnr = calculate_psnr(image_origin, image)
        ssim = calculate_ssim_piq(image_origin, image)
        print(f"PSNR={psnr:.2f}  SSIM={ssim:.4f}")
    else:
        parser.error("请提供 --recovered + --original（单图）或 --dir + --origin（批量）")