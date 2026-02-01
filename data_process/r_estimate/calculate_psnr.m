eps1_path = 'image1.png';
eps2_path = 'image2.png';

img1 = imread(eps1_path);
img2 = imread(eps2_path);

mse = mean((double(img1(:)) - double(img2(:))).^2);

if mse == 0
    psnr_value = Inf;
else
    psnr_value = 10 * log10(255^2 / mse);
end

fprintf('PSNR = %.2f dB\n', psnr_value);
fprintf('MSE = %.6f\n', mse);