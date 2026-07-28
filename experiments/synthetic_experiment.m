clc
clear
close all

% 路径自定位（可从任意目录运行）
this_dir = fileparts(mfilename('fullpath'));
repo_root = fileparts(this_dir);
figure_dir = fullfile(repo_root, 'figure');
text_dir = fullfile(repo_root, 'text');
if ~exist(figure_dir, 'dir'), mkdir(figure_dir); end
if ~exist(text_dir, 'dir'), mkdir(text_dir); end

m = 200; n =200; k_true = 10;
OS = 5;
max_iter = 200;
tol = 1e-8;

U_true = randn(m, k_true);
V_true = randn(n, k_true);
A_true = U_true * V_true';

dim_Mk = (m + n - k_true) * k_true;

num_observations = round(OS * dim_Mk);
[I, J] = ind2sub([m, n], randperm(m*n, num_observations));
Omega = false(m, n);
Omega(sub2ind([m, n], I, J)) = true;

A_observed = A_true .* Omega;

methods = { 'HZ', 'DY', 'FR', 'PRP', 'HS', 'NHS','Alg1'};
results = struct();

for i = 1:length(methods)
    method = methods{i};
    fprintf('Running method: %s\n', method);

    tic;
    [X_recovered, errors] = matrix_completion(A_observed, Omega, k_true, max_iter, tol, method);
    elapsed_time = toc;

    recovery_error = norm(X_recovered - A_true, 'fro') / norm(A_true, 'fro');

    results.(method).X_recovered = X_recovered;
    results.(method).errors = errors;
    results.(method).time = elapsed_time;
    results.(method).recovery_error = recovery_error;
    results.(method).iterations = length(errors);
end

figure('Position', [100, 100, 800, 600]);

colors = {'b', 'r', 'g', 'm', 'c', 'k', [0.85 0.3 0]};
markers = {'o', 's', 'd', '^', 'v', '>', 'p'};
linestyles = {'-', '--', '-.', ':};

for i = 1:length(methods)
    method = methods{i};

    iterations = 0:length(results.(method).errors)-1;
    errors = results.(method).errors;

    color = colors{mod(i-1, length(colors)) + 1};
    marker = markers{mod(i-1, length(markers)) + 1};
    linestyle = linestyles{mod(i-1, length(linestyles)) + 1};

    semilogy(iterations, errors, ...
        'Color', color, ...
        'Marker', marker, ...
        'LineStyle', linestyle, ...
        'LineWidth', 1.5, ...
        'MarkerSize', 4, ...
        'DisplayName', method);

    hold on;
end

ylim([1e-8 1e3]);
xlabel('Iteration');
ylabel('$\|\mathrm{grad}\|$', 'Interpreter', 'latex');

titleStr = sprintf('m=%d, n=%d, r=%d, OS=%d', m, n, k_true, OS);
title(titleStr, 'FontSize', 12);

lgd = legend('Location', 'best');
lgd.FontSize = 10;
lgd.Box = 'on';
lgd.EdgeColor = 'black';

filename_eps = fullfile(figure_dir, sprintf('n=%d_m=%d_r=%d_os=%d.eps', n, m, k_true, OS));
saveas(gcf, filename_eps, 'epsc');

filename_txt = fullfile(text_dir, sprintf('n=%d_m=%d_r=%d_os=%d.txt', n, m, k_true, OS));
fid = fopen(filename_txt, 'w');
fprintf('\n比较结果:\n');
fprintf(titleStr);
fprintf('%s\n', repmat('-', 1, 150));
fprintf('%-8s %-10s %-12s %-12s %-15s\n', '方法', '迭代次数', '时间(秒)', '重构误差', '最终梯度范数');
fprintf('%s\n', repmat('-', 1, 80));
for i = 1:length(methods)
    method = methods{i};
    res = results.(method);
    fprintf('%-8s %-10d %-12.4f %-12.6f %-15.2e\n', ...
        method, res.iterations, res.time, ...
        res.recovery_error, res.errors(end));
end
fprintf('%s\n', repmat('-', 1, 80));


fprintf(fid, '%-8s %-12s %-12s %-12s %-15s\n', '方法', '迭代次数', '时间(秒)', '重构误差', '最终梯度范数');
fprintf(fid, '%s\n', repmat('-', 1, 60));
for i = 1:length(methods)
    method = methods{i};
    res = results.(method);

    fprintf(fid, '%-8s %-12d %-12.4f %-12.6f %-15.2e\n', ...
        method, ...
        res.iterations, ...
        res.time, ...
        res.recovery_error, ...
        res.errors(end));
end

fprintf(fid, '%s\n', repmat('-', 1, 80));

disp('结果已保存到 results.txt');