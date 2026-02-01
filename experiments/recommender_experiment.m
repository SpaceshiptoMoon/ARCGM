clc
clear
close all

data = readmatrix('../data/rating_fillna.csv', 'NumHeaderLines', 1);
A_observed = data(1:610, 2:1000);
m = 610;
n = 1000;
k_true = 10;
max_iter = 50;
tol = 1e-8;
Omega = readmatrix('../data/rating_mask.csv', 'NumHeaderLines', 1);
Omega = Omega(1:610, 2:1000);
A_observed = A_observed .* Omega;

methods = { 'HZ', 'DY', 'FR', 'PRP', 'HS','NHS','Alg1'};
results = struct();

for i = 1:length(methods)
    method = methods{i};
    fprintf('Running method: %s\n', method);

    tic;
    [X_recovered, errors] = matrix_completion(...
        A_observed, Omega, k_true, max_iter, tol, method);
    elapsed_time = toc;

    MAE_history = errors(:,2);
    RMSE_history = errors(:,3);

    results.(method).MAE_history = MAE_history;
    results.(method).RMSE_history = RMSE_history;
    results.(method).time = elapsed_time;
    results.(method).iterations = length(errors);
end

line_styles = {'-', '--', ':', '-.', '--', ':', '-.'};
colors = lines(length(methods));

colors(end,:) = [1 0 1];
line_styles{end} = '--';
last_line_width = 3;

figure('Position', [100, 100, 800, 400], 'Color', 'w');
hold on; grid on; grid minor;
title('MAE Convergence Comparison', 'FontSize', 12, 'FontWeight', 'bold');
xlabel('Iteration', 'FontSize', 10);
ylabel('MAE Value', 'FontSize', 10);

all_MAE = cellfun(@(m) results.(m).MAE_history, methods, 'UniformOutput', false);
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
tableDir = '../results/table/';
if ~exist(tableDir, 'dir')
    mkdir(tableDir);
end
saveas(gcf, fullfile(tableDir, 'MAE_convergence_lines_only.eps'), 'epsc');
close;

figure('Position', [100, 100, 800, 400], 'Color', 'w');
hold on; grid on; grid minor;
title('RMSE Convergence Comparison', 'FontSize', 12, 'FontWeight', 'bold');
xlabel('Iteration', 'FontSize', 10);
ylabel('RMSE Value', 'FontSize', 10);

all_RMSE = cellfun(@(m) results.(m).RMSE_history, methods, 'UniformOutput', false);
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

tableDir = '../results/table/';
textDir = '../results/text/';
if ~exist(tableDir, 'dir')
    mkdir(tableDir);
end
if ~exist(textDir, 'dir')
    mkdir(textDir);
end

legend('show', 'Location', 'best', 'FontSize', 8);
tableDir = '../results/table/';
if ~exist(tableDir, 'dir')
    mkdir(tableDir);
end

saveas(gcf, fullfile(tableDir, 'RMSE_convergence_lines_only.eps'), 'epsc');

textDir = '../results/text/';
if ~exist(textDir, 'dir')
    mkdir(textDir);
end

filename_txt = fullfile(textDir, sprintf('n=%d_m=%d_r=%d_iterations.txt', n, m, k_true));

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