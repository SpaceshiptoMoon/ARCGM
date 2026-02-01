clc
data = readmatrix('../data/rating_fillna.csv', 'NumHeaderLines', 1);
m =  1000;
n = 1000;
A_observed = data(1:4000, 2:end);
k_true = 40;
max_iter = 1000;
tol = 1e-8;
Omega = readmatrix('../data/rating_mask.csv', 'NumHeaderLines', 1);
Omega = Omega(1:4000, 2:end);

A_observed = A_observed .* Omega;
methods = {'HZ', 'DY','FR','PRP', 'HS', 'NHS','Alg1'};
results = struct();

for i = 1:length(methods)
    method = methods{i};
    fprintf('Running method: %s\n', method);

    tic;
    [X_recovered, errors] = matrix_completion(A_observed, Omega, k_true, max_iter, tol, method);
    elapsed_time = toc;

    recovery_error_MAE = sum(sum(abs(A_observed - X_recovered.* Omega))) / sum(Omega(:));
    recovery_error_RMSE = norm(A_observed - X_recovered.* Omega, 'fro') / sqrt(sum(sum(Omega)));

    fprintf('Method: %s\n', method);
    fprintf('  MAE:    %.6f\n', recovery_error_MAE);
    fprintf('  RMSE:   %.6f\n', recovery_error_RMSE);

    results.(method).X_recovered = X_recovered;
    results.(method).errors = errors;
    results.(method).time = elapsed_time;
    results.(method).iterations = length(errors);
end

fig = figure('Position', [100, 100, 800, 600]);

colors = {'b', 'r', 'g', 'm', 'c', 'k', [0.85 0.3 0]};
markers = {'o', 's', 'd', '^', 'v', '>', 'p'};
linestyles = {'-', '--', '-.', ':'};

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

title('Convergence Comparison with Wolfe Line Search', 'FontSize', 14, 'FontWeight', 'normal');

lgd = legend('Location', 'best');
lgd.FontSize = 10;
lgd.Box = 'on';
lgd.EdgeColor = 'black';

filename = sprintf('../results/table/n=%d_m=%d_r=%d.eps', n, m, k_true);
print(fig, '-depsc', '-r300', '-painters', filename);
fprintf('已保存EPS图像到: %s\n', filename);

close(fig);

filename_txt = sprintf('../results/text/n=%d_m=%d_r=%d.txt', n, m, k_true);
fid = fopen(filename_txt, 'w');
fprintf('\n比较结果:\n');
fprintf('Convergence Comparison with Wolfe Line Search\n');
fprintf('%s\n', repmat('-', 1, 150));
fprintf('%-8s %-10s %-12s %-12s %-15s\n', '方法', '迭代次数', '时间(秒)', '重构误差', '最终梯度范数');
fprintf('%s\n', repmat('-', 1, 80));
for i = 1:length(methods)
    method = methods{i};
    res = results.(method);
    fprintf('%-8s %-10d %-12.4f  %-15.2e\n', ...
        method, res.iterations, res.time, ...
         res.errors(end));
end
fprintf('%s\n', repmat('-', 1, 80));

fprintf(fid, '%-8s %-12s %-12s %-12s %-15s\n', '方法', '迭代次数', '时间(秒)', '重构误差', '最终梯度范数');
fprintf(fid, '%s\n', repmat('-', 1, 60));
for i = 1:length(methods)
    method = methods{i};
    res = results.(method);

    fprintf(fid, '%-8s %-12d %-12.4f %-15.2e\n', ...
        method, ...
        res.iterations, ...
        res.time, ...
        res.errors(end));
end

fprintf(fid, '%s\n', repmat('-', 1, 80));

fclose(fid);
disp('结果已保存到文本文件');