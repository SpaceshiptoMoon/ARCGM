function psnr_value = calculate_psnr(path1, path2)
%CALCULATE_PSNR 计算两幅图像之间的 PSNR（dB）。
%
%   v = calculate_psnr(path1, path2)
%
% 输入为两幅图像文件路径（8 位基准 255）。

    if nargin < 2
        error('用法: calculate_psnr(path1, path2)');
    end
    img1 = imread(path1);
    img2 = imread(path2);

    mse = mean((double(img1(:)) - double(img2(:))).^2);
    if mse == 0
        psnr_value = Inf;
    else
        psnr_value = 10 * log10(255^2 / mse);
    end

    fprintf('PSNR = %.2f dB\n', psnr_value);
    fprintf('MSE = %.6f\n', mse);
end