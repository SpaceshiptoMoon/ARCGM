from PIL import Image
import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image
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
    # dir_path = r"C:\Users\xl\Desktop\论文二\results\rank80\os0.7"
    # image_path = r"C:\Users\xl\Desktop\论文二\results\rank80\rank_80_image.png"
    # writer_txt(dir_path, image_path)
    image = Image.open(r"C:\Users\xl\Desktop\论文二\results\rank50\block100\HZ.png") 
    image_origin = (np.array(Image.open(r"C:\Users\xl\Desktop\论文二\results\rank50\rank_50_image.png")) / 65535.0 * 255).astype(np.uint8)
    psnr = calculate_psnr(image_origin, np.array(image))
    ssim = calculate_ssim_piq(image_origin, np.array(image))
    print(psnr, ssim)