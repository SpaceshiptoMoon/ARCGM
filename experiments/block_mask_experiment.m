clc
clear
close all

% 路径自定位（可从任意目录运行）
this_dir = fileparts(mfilename('fullpath'));
repo_root = fileparts(this_dir);
results_dir = fullfile(repo_root, 'results');

% 实验参数
rank = 50;
block_size = 100;
max_iter = 90;
tol = 1e-12;
methods = {'HZ'};   % 块缺失实验目前只演示 HZ，可按需扩展为 7 种

% 读取低秩图（由 data_process/image/create_lowrank_image.m(rank) 生成）
rank_dir = fullfile(results_dir, sprintf('rank%d', rank));
img_path = fullfile(rank_dir, sprintf('rank_%d_image.png', rank));
fprintf('读取低秩图像: %s\n', img_path);

img_original = imread(img_path);
if size(img_original, 3) > 1
    img_original = rgb2gray(img_original);
end
img_original = im2double(img_original);

% 中心块缺失掩膜
[h, w] = size(img_original);
x_start = floor((w - block_size)/2);
y_start = floor((h - block_size)/2);
mask = true(size(img_original));
mask(y_start:y_start+block_size-1, x_start:x_start+block_size-1) = false;
img_damaged = img_original .* mask;

save_dir = fullfile(rank_dir, sprintf('block%d', block_size));
if ~exist(save_dir, 'dir'), mkdir(save_dir); end

all_errors = cell(length(methods), 1);
all_psnr = zeros(length(methods), 1);
all_recovered = cell(length(methods), 1);
all_times = zeros(length(methods), 1);
all_ssim = zeros(length(methods), 1);

for i = 1:length(methods)
    method = methods{i};
    fprintf('\n正在运行方法: %s...\n', method);

    tic;
    [img_recovered, errors] = matrix_completion(...
        img_damaged, mask, rank, max_iter, tol, method);
    elapsed_time = toc;

    all_times(i) = elapsed_time;
    fprintf('方法 %s 运行时间: %.4f 秒\n', method, elapsed_time);

    all_errors{i} = errors;
    all_psnr(i) = psnr(img_recovered, img_original);
    all_recovered{i} = img_recovered;
    [all_ssim(i), ssim_map] = ssim(img_recovered, img_original);

    f = figure('Visible', 'off');
    imshow(img_recovered);
    set(gcf, 'PaperPositionMode', 'auto');
    print(f, fullfile(save_dir, sprintf('%s.eps', method)), '-depsc', '-r300');
    close(f);
    imwrite(img_recovered, fullfile(save_dir, sprintf('%s.png', method)));
    fprintf('已保存恢复图像(EPS/PNG): %s\n', fullfile(save_dir, method));
end

% 收敛曲线
fig_curve = figure('Visible', 'off');
hold on;
colors = lines(length(methods));
for i = 1:length(methods)
    semilogy(all_errors{i}, 'LineWidth', 2, 'DisplayName', methods{i}, 'Color', colors(i,:));
end
hold off;
set(gca, 'YScale', 'log');
xlabel('Iteration');
ylabel('error');
title('方法对比收敛曲线');
legend('show', 'Location', 'best');
grid on;
set(gcf, 'PaperPositionMode', 'auto');
print(fig_curve, fullfile(save_dir, 'convergence_comparison_log.eps'), '-depsc', '-r300');
saveas(fig_curve, fullfile(save_dir, 'convergence_comparison_log.png'));
close(fig_curve);

% 损坏图像
f = figure('Visible', 'off');
imshow(img_damaged);
set(gcf, 'PaperPositionMode', 'auto');
print(f, fullfile(save_dir, 'damaged_image.eps'), '-depsc', '-r300');
close(f);
imwrite(img_damaged, fullfile(save_dir, 'damaged_image.png'));

% 性能结果表
fid = fopen(fullfile(save_dir, 'performance_results.txt'), 'w');
fprintf(fid, '性能结果比较:\n');
for i = 1:length(methods)
    fprintf(fid, '%5s方法: PSNR=%.2f dB SSIM=%.2f| 时间=%.4f秒\n', ...
        methods{i}, all_psnr(i), all_ssim(i), all_times(i));
end
fclose(fid);

% 汇总图（3x3 子图）
fig_summary = figure('Visible', 'off');
subplot(3, 3, 1); imshow(img_original); title('原始图像');
subplot(3, 3, 2); imshow(img_damaged); title('损坏图像');
for i = 1:length(methods)
    subplot(3, 3, i+2);
    imshow(all_recovered{i});
    title(sprintf('%s (%.2f dB)', methods{i}, all_psnr(i)));
end
set(gcf, 'PaperPositionMode', 'auto');
print(fig_summary, fullfile(save_dir, 'summary_results.eps'), '-depsc', '-r300');
saveas(fig_summary, fullfile(save_dir, 'summary_results.png'));
close(fig_summary);

fprintf('\n所有结果已保存到目录: %s\n', save_dir);