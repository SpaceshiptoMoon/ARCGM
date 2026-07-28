clc
clear
close all

% 路径自定位（可从任意目录运行）
this_dir = fileparts(mfilename('fullpath'));
repo_root = fileparts(this_dir);
results_dir = fullfile(repo_root, 'results');

% 实验参数
rank = 80;
os = 0.7;          % 缺失比例：保留 (1-os) 的像素作为观测
max_iter = 100;
tol = 1e-12;
methods = {'HZ', 'DY', 'FR', 'PRP', 'HS', 'NHS', 'Alg1'};

% 读取低秩图（由 data_process/image/create_lowrank_image.m(rank) 生成）
rank_dir = fullfile(results_dir, sprintf('rank%d', rank));
img_path = fullfile(rank_dir, sprintf('rank_%d_image.png', rank));
fprintf('读取低秩图像: %s\n', img_path);

img_original = imread(img_path);
if size(img_original, 3) > 1
    img_original = rgb2gray(img_original);
end
img_original = im2double(img_original);

rng(123);
mask = rand(size(img_original)) > os;     % true = 缺失
img_damaged = img_original .* mask;

save_dir = fullfile(rank_dir, sprintf('os%.1f', os));
if ~exist(save_dir, 'dir')
    mkdir(save_dir);
end

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

    fig = figure('Position', [100, 100, 512, 512]);
    imshow(img_recovered, []);
    axis off;
    set(gcf, 'Color', 'white');

    recovered_path_eps = fullfile(save_dir, sprintf('%s.eps', method));
    print(fig, '-depsc', '-r300', recovered_path_eps);
    fprintf('已保存EPS恢复图像到: %s\n', recovered_path_eps);

    recovered_path_png = fullfile(save_dir, sprintf('%s.png', method));
    imwrite(img_recovered, recovered_path_png);
    fprintf('已保存PNG恢复图像到: %s\n', recovered_path_png);

    close(fig);
end

fig_curve = figure('Position', [100, 100, 800, 600]);
hold on;
colors = lines(length(methods));
for i = 1:length(methods)
    semilogy(all_errors{i}, 'LineWidth', 2, 'DisplayName', methods{i}, 'Color', colors(i,:));
end
hold off;
set(gca, 'YScale', 'log');
xlabel('Iteration');
ylabel('error');
grid on;
legend('Location', 'best');

curve_path_eps = fullfile(save_dir, 'convergence_comparison_log.eps');
print(fig_curve, '-depsc', '-r300', curve_path_eps);
fprintf('已保存对数坐标对比曲线(EPS)到: %s\n', curve_path_eps);

curve_path_png = fullfile(save_dir, 'convergence_comparison_log.png');
saveas(fig_curve, curve_path_png);
fprintf('已保存对数坐标对比曲线(PNG)到: %s\n', curve_path_png);

close(fig_curve);

damaged_path_eps = fullfile(save_dir, 'damaged_image.eps');
fig_damaged = figure('Position', [100, 100, 512, 512]);
imshow(img_damaged, []);
axis off;
set(gcf, 'Color', 'white');
print(fig_damaged, '-depsc', '-r300', damaged_path_eps);
close(fig_damaged);
fprintf('已保存损坏图像(EPS)到: %s\n', damaged_path_eps);

damaged_path_png = fullfile(save_dir, 'damaged_image.png');
imwrite(img_damaged, damaged_path_png);
fprintf('已保存损坏图像(PNG)到: %s\n', damaged_path_png);

result_txt_path = fullfile(save_dir, 'performance_results.txt');
fid = fopen(result_txt_path, 'w');
fprintf(fid, '性能结果比较:\n');
fprintf('\n性能结果比较:\n');
for i = 1:length(methods)
    fprintf(fid, '%5s方法: PSNR=%.2f dB SSIM=%.4f| 时间=%.4f秒\n', methods{i}, all_psnr(i), all_ssim(i), all_times(i));
    fprintf('%5s方法: PSNR=%.2f dB SSIM=%.4f| 时间=%.4f秒\n', methods{i}, all_psnr(i), all_ssim(i), all_times(i));
end
fclose(fid);
fprintf('已保存性能结果到: %s\n', result_txt_path);

fig_summary = figure('Position', [100, 100, 1200, 900]);
subplot(3, 3, 1);
imshow(img_original, []);
axis off;
title('Original Image', 'FontSize', 10);

subplot(3, 3, 2);
imshow(img_damaged, []);
axis off;
title('Damaged Image', 'FontSize', 10);

for i = 1:length(methods)
    subplot(3, 3, i+2);
    imshow(all_recovered{i}, []);
    axis off;
    title(sprintf('%s Recovery', methods{i}), 'FontSize', 10);
end

summary_path_eps = fullfile(save_dir, 'summary_results.eps');
print(fig_summary, '-depsc', '-r300', summary_path_eps);
fprintf('已保存结果汇总图(EPS)到: %s\n', summary_path_eps);

summary_path_png = fullfile(save_dir, 'summary_results.png');
saveas(fig_summary, summary_path_png);
fprintf('已保存结果汇总图(PNG)到: %s\n', summary_path_png);

close(fig_summary);

fprintf('\n所有结果已保存到目录: %s\n', save_dir);