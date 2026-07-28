clc
clear
close all

% 路径自定位（可从任意目录运行）
this_dir = fileparts(mfilename('fullpath'));
repo_root = fileparts(this_dir);
data_dir = fullfile(repo_root, 'data');
results_dir = fullfile(repo_root, 'results');

% 实验参数
m = 610;          % 用户数
n = 1000;         % 电影数（取评分最多的 top 1000）
k_true = 10;
max_iter = 50;
tol = 1e-8;
methods = {'HZ', 'DY', 'FR', 'PRP', 'HS', 'NHS', 'Alg1'};

% 读取 MovieLens 预处理产物（由 data_process/recommendation_system/movie_rating_processor.py 生成）
rating_path = fullfile(data_dir, 'rating_fillna.csv');
mask_path = fullfile(data_dir, 'rating_mask.csv');
data = readmatrix(rating_path, 'NumHeaderLines', 1);
A_observed = data(1:m, 2:n+1);
Omega = readmatrix(mask_path, 'NumHeaderLines', 1);
Omega = logical(Omega(1:m, 2:n+1));
A_observed = A_observed .* Omega;

results = struct();
for i = 1:length(methods)
    method = methods{i};
    fprintf('Running method: %s\n', method);

    tic;
    [X_recovered, errors, metrics] = matrix_completion(...
        A_observed, Omega, k_true, max_iter, tol, method);
    elapsed_time = toc;

    results.(method).MAE_history = metrics.MAE;
    results.(method).RMSE_history = metrics.RMSE;
    results.(method).grad_history = errors;
    results.(method).time = elapsed_time;
    results.(method).iterations = length(errors);
end

% 绘图样式
line_styles = {'-', '--', ':', '-.', '--', ':', '-.'};
colors = lines(length(methods));
colors(end,:) = [1 0 1];
line_styles{end} = '--';
last_line_width = 3;

% 输出目录
table_dir = fullfile(results_dir, 'table');
text_dir = fullfile(results_dir, 'text');
if ~exist(table_dir, 'dir'), mkdir(table_dir); end
if ~exist(text_dir, 'dir'), mkdir(text_dir); end

% --- MAE 收敛曲线 ---
fig_mae = figure('Position', [100, 100, 800, 400], 'Color', 'w');
hold on; grid on; grid minor;
title('MAE Convergence Comparison', 'FontSize', 12, 'FontWeight', 'bold');
xlabel('Iteration', 'FontSize', 10);
ylabel('MAE Value', 'FontSize', 10);

all_MAE = cellfun(@(mth) results.(mth).MAE_history, methods, 'UniformOutput', false);
min_val = min(cellfun(@min, all_MAE));
max_val = max(cellfun(@max, all_MAE));
ylim([min_val * 0.8, max_val * 1.1]);
yticks(linspace(min_val, max_val, 8));
ytickformat('%.2f');

for i = 1:length(methods)
    method = methods{i};
    if i == length(methods)
        plot(1:length(results.(method).MAE_history), results.(method).MAE_history, ...
            line_styles{i}, 'Color', colors(i,:), 'LineWidth', last_line_width, 'DisplayName', method);
    else
        plot(1:length(results.(method).MAE_history), results.(method).MAE_history, ...
            line_styles{i}, 'Color', colors(i,:), 'LineWidth', 2, 'DisplayName', method);
    end
end
legend('show', 'Location', 'best', 'FontSize', 8);
print(fig_mae, '-depsc', '-r300', fullfile(table_dir, 'MAE_convergence_lines_only.eps'));
close(fig_mae);

% --- RMSE 收敛曲线 ---
fig_rmse = figure('Position', [100, 100, 800, 400], 'Color', 'w');
hold on; grid on; grid minor;
title('RMSE Convergence Comparison', 'FontSize', 12, 'FontWeight', 'bold');
xlabel('Iteration', 'FontSize', 10);
ylabel('RMSE Value', 'FontSize', 10);

all_RMSE = cellfun(@(mth) results.(mth).RMSE_history, methods, 'UniformOutput', false);
min_val = min(cellfun(@min, all_RMSE));
max_val = max(cellfun(@max, all_RMSE));
ylim([min_val * 0.8, max_val * 1.1]);
yticks(linspace(min_val, max_val, 8));
ytickformat('%.2f');

for i = 1:length(methods)
    method = methods{i};
    if i == length(methods)
        plot(1:length(results.(method).RMSE_history), results.(method).RMSE_history, ...
            line_styles{i}, 'Color', colors(i,:), 'LineWidth', last_line_width, 'DisplayName', method);
    else
        plot(1:length(results.(method).RMSE_history), results.(method).RMSE_history, ...
            line_styles{i}, 'Color', colors(i,:), 'LineWidth', 2, 'DisplayName', method);
    end
end
legend('show', 'Location', 'best', 'FontSize', 8);
print(fig_rmse, '-depsc', '-r300', fullfile(table_dir, 'RMSE_convergence_lines_only.eps'));
close(fig_rmse);

% --- 结果表 ---
filename_txt = fullfile(text_dir, sprintf('n=%d_m=%d_r=%d_iterations.txt', n, m, k_true));
fid = fopen(filename_txt, 'w');
if fid == -1
    error('无法创建文件，请检查路径权限或文件名是否合法。');
end

fprintf(fid, '%-8s %-12s %-12s %-15s %-15s\n', 'Method', 'Iterations', 'Time(s)', 'Final MAE', 'Final RMSE');
fprintf(fid, '%s\n', repmat('-', 1, 60));
for i = 1:length(methods)
    method = methods{i};
    res = results.(method);
    fprintf(fid, '%-8s %-12d %-12.4f %-15.6f %-15.6f\n', ...
        method, res.iterations, res.time, res.MAE_history(end), res.RMSE_history(end));
end
fclose(fid);
disp(['Results saved to ' filename_txt]);