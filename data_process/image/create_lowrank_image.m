img = imread('../data/test_image.png');
if size(img,3)==3
    img = rgb2gray(img);
end
img = im2double(img);

[U, S, V] = svd(img);

target_rank = 60;

U_trunc = U(:, 1:target_rank);
S_trunc = S(1:target_rank, 1:target_rank);
V_trunc = V(:, 1:target_rank);
low_rank_img = U_trunc * S_trunc * V_trunc';

base_dir = '../results/';
rank_dir = fullfile(base_dir, sprintf('rank%d', target_rank));

if ~exist(rank_dir, 'dir')
    mkdir(rank_dir);
    fprintf('已创建文件夹: %s\n', rank_dir);
end

output_path_eps = fullfile(rank_dir, sprintf('rank_%d_image.eps', target_rank));

figure('Visible', 'off');
imshow(low_rank_img);
print(output_path_eps, '-depsc', '-r300');
close(gcf);
fprintf('EPS格式图像已保存到: %s\n', output_path_eps);

output_path_png_lossless = fullfile(rank_dir, sprintf('rank_%d_image.png', target_rank));
img_uint16 = im2uint16(low_rank_img);
imwrite(img_uint16, output_path_png_lossless, 'PNG', 'BitDepth', 16);
fprintf('无损PNG图像已保存到: %s\n', output_path_png_lossless);

original_png_path = fullfile(rank_dir, 'original_grayscale.png');
imwrite(im2uint16(img), original_png_path, 'PNG', 'BitDepth', 16);
fprintf('原始灰度图（PNG）已保存到: %s\n', original_png_path);

fprintf('\n所有图像格式保存完成！\n');