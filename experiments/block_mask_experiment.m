rank = 50;
block_size = 100;
base_dir = '../results/';
rank_dir = fullfile(base_dir, sprintf('rank%d', rank));
output_filename = sprintf('rank_%d_image.png', rank);
img_path = fullfile(rank_dir, output_filename);
fprintf('读取低秩图像: %s\n', img_path);
max_iter = 90;
tol = 1e-12;
methods = {'HZ'};

img_original = imread(img_path);
if size(img_original, 3) > 1
    img_original = rgb2gray(img_original);
end
img_original = im2double(img_original);

[h, w] = size(img_original);
x_start = floor((w - block_size)/2);
y_start = floor((h - block_size)/2)-130;

mask = true(size(img_original));
mask(y_start:y_start+block_size-1, x_start:x_start+block_size-1) = false;
img_damaged = img_original .* mask;

save_dir = fullfile(rank_dir, sprintf('block%d', block_size));
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

    recovered_path_eps = fullfile(save_dir, sprintf('%s.eps', method));

    f = figure('Visible', 'off');
    imshow(img_recovered);
    set(gcf, 'PaperPositionMode', 'auto');
    print(f, recovered_path_eps, '-depsc', '-r300');
    close(f);

    fprintf('已保存EPS恢复图像到: %s\n', recovered_path_eps);

    recovered_path_png = fullfile(save_dir, sprintf('%s.png', method));
    imwrite(img_recovered, recovered_path_png);
    fprintf('已保存PNG恢复图像到: %s\n', recovered_path_png);
end

figure('Visible', 'off');
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

curve_path_eps = fullfile(save_dir, 'convergence_comparison_log.eps');
set(gcf, 'PaperPositionMode', 'auto');
print('-depsc', curve_path_eps, '-r300');
fprintf('已保存对数坐标对比曲线(EPS)到: %s\n', curve_path_eps);

curve_path_png = fullfile(save_dir, 'convergence_comparison_log.png');
saveas(gcf, curve_path_png);
fprintf('已保存对数坐标对比曲线(PNG)到: %s\n', curve_path_png);

close(gcf);

damaged_path_eps = fullfile(save_dir, 'damaged_image.eps');
f = figure('Visible', 'off');
imshow(img_damaged);
set(gcf, 'PaperPositionMode', 'auto');
print(f, damaged_path_eps, '-depsc', '-r300');
close(f);
fprintf('已保存损坏图像(EPS)到: %s\n', damaged_path_eps);

damaged_path_png = fullfile(save_dir, 'damaged_image.png');
imwrite(img_damaged, damaged_path_png);
fprintf('已保存损坏图像(PNG)到: %s\n', damaged_path_png);

result_txt_path = fullfile(save_dir, 'performance_results.txt');
fid = fopen(result_txt_path, 'w');
fprintf(fid, '性能结果比较:\n');
fprintf('\n性能结果比较:\n');
for i = 1:length(methods)
    fprintf(fid, '%5s方法: PSNR=%.2f dB SSIM=%.2f| 时间=%.4f秒\n', methods{i}, all_psnr(i), all_ssim(i), all_times(i));
    fprintf('%5s方法: PSNR=%.2f dB SSIM=%.2f| 时间=%.4f秒\n', methods{i}, all_psnr(i), all_ssim(i), all_times(i));
end
fclose(fid);
fprintf('已保存性能结果到: %s\n', result_txt_path);

figure('Visible', 'off');
subplot(3, 3, 1); imshow(img_original); title('原始图像');
subplot(3, 3, 2); imshow(img_damaged); title('损坏图像');
for i = 1:length(methods)
    subplot(3, 3, i+2);
    imshow(all_recovered{i});
    title(sprintf('%s (%.2f dB)', methods{i}, all_psnr(i)));
end

summary_path_eps = fullfile(save_dir, 'summary_results.eps');
set(gcf, 'PaperPositionMode', 'auto');
print('-depsc', summary_path_eps, '-r300');
fprintf('已保存结果汇总图(EPS)到: %s\n', summary_path_eps);

summary_path_png = fullfile(save_dir, 'summary_results.png');
saveas(fig_summary, summary_path_png);
fprintf('已保存结果汇总图(PNG)到: %s\n', summary_path_png);

close(gcf);

fprintf('\n所有结果已保存到目录: %s\n', save_dir);