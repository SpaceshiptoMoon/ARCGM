clc
clear
close all

% 路径自定位（可从任意目录运行）
this_dir = fileparts(mfilename('fullpath'));
repo_root = fileparts(this_dir);
data_dir = fullfile(repo_root, 'data');
results_dir = fullfile(repo_root, 'results');

% 实验参数
k_true = 40;
max_iter = 1000;
tol = 1e-8;
methods = {'HZ', 'DY', 'FR', 'PRP', 'HS', 'NHS', 'Alg1'};

% 读取 PEMS 交通数据预处理产物（由 data_process/traffic/PEM_data_process.py 生成）
% PE_data.csv：流量矩阵，缺失位置为 0；PE_mask.csv：观测掩码，1=观测 0=缺失
data = readmatrix(fullfile(data_dir, 'PE_data.csv'));
Omega = readmatrix(fullfile(data_dir, 'PE_mask.csv'));
Omega = logical(Omega);
A_observed = data .* Omega;
[m, n] = size(A_observed);
fprintf('PEMS 数据规模: m=%d, n=%d, 观测率=%.2f%%\n', m, n, 100*sum(Omega(:))/numel(Omega));

results = struct();
for i = 1:length(methods)
    method = methods{i};
    fprintf('Running method: %s\n', method);

    tic;
    [X_recovered, errors] = matrix_completion(A_observed, Omega, k_true, max_iter, tol, method);
    elapsed_time = toc;

    % 观测集上的 MAE / RMSE
    recovery_error_MAE = sum(sum(abs(A_observed - X_recovered .* Omega))) / sum(Omega(:));
    recovery_error_RMSE = norm(A_observed - X_recovered .* Omega, 'fro') / sqrt(sum(Omega(:)));

    fprintf('Method: %s\n', method);
    fprintf('  MAE:    %.6f\n', recovery_error_MAE);
    fprintf('  RMSE:   %.6f\n', recovery_error_RMSE);

    results.(method).X_recovered = X_recovered;
    results.(method).errors = errors;
    results.(method).time = elapsed_time;
    results.(method).MAE = recovery_error_MAE;
    results.(method).RMSE = recovery_error_RMSE;
    results.(method).iterations = length(errors);
end

% 输出目录
table_dir = fullfile(results_dir, 'table');
text_dir = fullfile(results_dir, 'text');
if ~exist(table_dir, 'dir'), mkdir(table_dir); end
if ~exist(text_dir, 'dir'), mkdir(text_dir); end

% --- 梯度范数收敛曲线 ---
fig = figure('Position', [100, 100, 800, 600]);
colors = {'b', 'r', 'g', 'm', 'c', 'k', [0.85 0.3 0]};
markers = {'o', 's', 'd', '^', 'v', '>', 'p'};
linestyles = {'-', '--', '-.', ':'};

hold on;
for i = 1:length(methods)
    method = methods{i};
    iterations = 0:length(results.(method).errors)-1;
    semilogy(iterations, results.(method).errors, ...
        'Color', colors{mod(i-1, length(colors)) + 1}, ...
        'Marker', markers{mod(i-1, length(markers)) + 1}, ...
        'LineStyle', linestyles{mod(i-1, length(linestyles)) + 1}, ...
        'LineWidth', 1.5, 'MarkerSize', 4, 'DisplayName', method);
end
hold off;
ylim([1e-8 1e3]);
xlabel('Iteration');
ylabel('$\|\mathrm{grad}\|$', 'Interpreter', 'latex');
title('Convergence Comparison with Wolfe Line Search', 'FontSize', 14);
lgd = legend('Location', 'best');
lgd.FontSize = 10;
lgd.Box = 'on';
lgd.EdgeColor = 'black';

filename_eps = fullfile(table_dir, sprintf('n=%d_m=%d_r=%d.eps', n, m, k_true));
print(fig, '-depsc', '-r300', '-painters', filename_eps);
fprintf('已保存EPS图像到: %s\n', filename_eps);
close(fig);

% --- 结果表 ---
filename_txt = fullfile(text_dir, sprintf('n=%d_m=%d_r=%d.txt', n, m, k_true));
fid = fopen(filename_txt, 'w');
if fid == -1
    error('无法创建文件: %s', filename_txt);
end
fprintf('Convergence Comparison with Wolfe Line Search\n');
fprintf('%-8s %-12s %-12s %-12s %-15s\n', '方法', '迭代次数', '时间(秒)', 'MAE', '最终梯度范数');
fprintf('%s\n', repmat('-', 1, 80));
fprintf(fid, '%-8s %-12s %-12s %-12s %-15s\n', '方法', '迭代次数', '时间(秒)', 'MAE', '最终梯度范数');
fprintf(fid, '%s\n', repmat('-', 1, 80));
for i = 1:length(methods)
    method = methods{i};
    res = results.(method);
    fprintf('%-8s %-12d %-12.4f %-12.6f %-15.2e\n', method, res.iterations, res.time, res.MAE, res.errors(end));
    fprintf(fid, '%-8s %-12d %-12.4f %-12.6f %-15.2e\n', method, res.iterations, res.time, res.MAE, res.errors(end));
end
fprintf('%s\n', repmat('-', 1, 80));
fprintf(fid, '%s\n', repmat('-', 1, 80));
fclose(fid);
disp(['结果已保存到 ' filename_txt]);