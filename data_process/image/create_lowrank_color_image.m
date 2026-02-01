img = imread('../data/test_color_image.png');
img = im2double(img);

figure;
subplot(1,2,1); imshow(img); title('原始彩色图像');

target_rank = 80;
low_rank_img = zeros(size(img));

for channel = 1:3
    [U, S, V] = svd(img(:,:,channel));

    U_trunc = U(:, 1:target_rank);
    S_trunc = S(1:target_rank, 1:target_rank);
    V_trunc = V(:, 1:target_rank);

    low_rank_img(:,:,channel) = U_trunc * S_trunc * V_trunc';
end

subplot(1,2,2); imshow(low_rank_img);
title(['秩为',num2str(target_rank),'的低秩彩色图像']);

base_dir = '../results/color/';
rank_dir = fullfile(base_dir, sprintf('rank%d', target_rank));

if ~exist(rank_dir, 'dir')
    mkdir(rank_dir);
    fprintf('已创建文件夹: %s\n', rank_dir);
end

output_path = fullfile(rank_dir, sprintf('rank_%d_image.png', target_rank));
imwrite(low_rank_img, output_path);
fprintf('低秩彩色图像已保存到: %s\n', output_path);