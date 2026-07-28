function [X, errors, metrics] = matrix_completion(A_observed, Omega, k, max_iter, tol, method)
%MATRIX_COMPLETION 加速黎曼共轭梯度法求解低秩矩阵补全问题。
%
%   [X, errors] = matrix_completion(A_observed, Omega, k, max_iter, tol, method)
%   [X, errors, metrics] = matrix_completion(...)  % 额外返回逐迭代 MAE/RMSE
%
% 输入：
%   A_observed - 观测矩阵（未观测位置置 0）
%   Omega      - 观测掩码，logical 矩阵，true 表示该位置已观测
%   k          - 目标低秩
%   max_iter   - 最大迭代次数
%   tol        - 梯度范数收敛阈值
%   method     - CG 变体：'HZ' | 'DY' | 'FR' | 'PRP' | 'HS' | 'NHS' | 'Alg1'
%
% 输出：
%   X       - 补全后的稠密矩阵
%   errors  - 逐迭代黎曼梯度 Frobenius 范数（单列向量）
%   metrics - 可选，struct，字段：
%             .MAE  - 逐迭代观测集上的 MAE
%             .RMSE - 逐迭代观测集上的 RMSE
%
% 线搜索参数：rho=1e-4（Armijo），sigma=0.6（曲率）。
% 流形运算：retraction（SVD 回撤）、vector_transport（投影式向量传输）、
%           compute_riemannian_gradient（切空间投影）、P_Omega（观测投影）。

    rho = 1e-4;
    sigma = 0.6;
    alpha_init = 1;

    [m, n] = size(A_observed);

    [U0, S0, V0] = svds(A_observed, k);
    X.U = U0;
    X.S = S0;
    X.V = V0;

    grad_prev = compute_riemannian_gradient(X, A_observed, Omega);
    eta_prev = -grad_prev;
    f_prev = objective_function(X, A_observed, Omega);

    errors = zeros(max_iter, 1);
    mae = zeros(max_iter, 1);
    rmse = zeros(max_iter, 1);


   for t = 1:max_iter

        errors(t) = norm(grad_prev, 'fro');

        % 观测集上的逐迭代 MAE / RMSE（供 recommender 等需要误差收敛曲线的实验）
        X_full_curr = X.U * X.S * X.V';
        resid_curr = P_Omega(X_full_curr - A_observed, Omega);
        mae(t) = sum(abs(resid_curr(:))) / sum(Omega(:));
        rmse(t) = norm(resid_curr, 'fro') / sqrt(sum(Omega(:)));

        alpha = wolfe_line_search(X, eta_prev, A_observed, Omega, k, ...
                                 f_prev, grad_prev, rho, sigma, alpha_init);


        switch method
            case 'Alg1'
                Z = retraction(X, alpha * eta_prev, k);

                grad_X = compute_riemannian_gradient(X, A_observed, Omega);
                grad_Z = compute_riemannian_gradient(Z, A_observed, Omega);
                transported_grad = vector_transport(Z, grad_X);
                Y_k = transported_grad - grad_Z;

                a_k = alpha * sum(grad_X(:) .* eta_prev(:));
                transported_eta = vector_transport(Z, eta_prev);
                b_k = -alpha * sum(Y_k(:) .* transported_eta(:));

                if abs(b_k) > 1e-10
                    gamma_k = -a_k / b_k;
                    X_new = retraction(X, gamma_k * alpha * eta_prev, k);
                else
                    X_new = Z;
                end

                grad = compute_riemannian_gradient(X_new, A_observed, Omega);
                f_new = objective_function(X_new, A_observed, Omega);

                eta_transported = vector_transport(X_new, eta_prev);

                beta = compute_beta(X_new, alpha, grad, grad_prev, eta_prev, 'Alg1');

                eta = -grad + beta * eta_transported;


            otherwise

                X_new = retraction(X, alpha * eta_prev, k);
                grad = compute_riemannian_gradient(X_new, A_observed, Omega);
                f_new = objective_function(X_new, A_observed, Omega);

                eta_transported = vector_transport(X_new, eta_prev);

                beta = compute_beta(X_new,alpha, grad, grad_prev, eta_prev, method);
                eta = -grad + beta * eta_transported;
        end

        X = X_new;
        grad_prev = grad;
        eta_prev = eta;
        f_prev = f_new;


        if errors(t) <= tol
            fprintf('收敛于迭代 %d，残差: %.4e\n', t, errors(t));
            errors = errors(1:t);
            mae = mae(1:t);
            rmse = rmse(1:t);
            break;
        end

        if t == max_iter
            fprintf('达到最大迭代次数，残差: %.4e\n', errors(t));
        end
    end

    metrics.MAE = mae(1:numel(errors));
    metrics.RMSE = rmse(1:numel(errors));

    X = X.U * X.S * X.V';
end

function alpha = wolfe_line_search(X, eta, A_observed, Omega, k, f, grad, rho, sigma, alpha_init)
    alpha = alpha_init;
    alpha_min = 0;
    alpha_max = inf;
    max_iter = 20;
    iter = 0;

    while iter < max_iter
        X_new = retraction(X, alpha * eta, k);
        f_new = objective_function(X_new, A_observed, Omega);

        grad_inner = sum(grad(:) .* eta(:));

        if f_new > f + rho * alpha * grad_inner
            alpha_max = alpha;
            alpha = (alpha_min + alpha_max)/2;
            iter = iter + 1;
            continue;
        end

        grad_new = compute_riemannian_gradient(X_new, A_observed, Omega);
        transported_eta = vector_transport(X_new, eta);
        grad_new_inner = sum(grad_new(:) .* transported_eta(:));

        if grad_new_inner < sigma * grad_inner
            alpha_min = alpha;
            if isinf(alpha_max)
                alpha = alpha * 2;
            else
                alpha = (alpha_min + alpha_max)/2;
            end
        else
            break;
        end

        iter = iter + 1;
    end
end

function f = objective_function(X, A_observed, Omega)
    residual = P_Omega(X.U * X.S * X.V' - A_observed, Omega);
    f = 0.5 * norm(residual, 'fro')^2;
end

function grad = compute_riemannian_gradient(X, A, Omega)
    X_full = X.U * X.S * X.V';
    euc_grad = P_Omega(X_full - A, Omega);

    PU = X.U * X.U';
    PV = X.V * X.V';
    PU_perp = eye(size(PU)) - PU;
    PV_perp = eye(size(PV)) - PV;

    grad = PU * euc_grad * PV_perp + PU_perp * euc_grad * PV + PU * euc_grad * PV;
end

function X_new = retraction(X, eta, k)
    X_full = X.U * X.S * X.V' + eta;
    [U_new, S_new, V_new] = svds(X_full, k);
    X_new.U = U_new;
    X_new.S = S_new;
    X_new.V = V_new;
end

function transported_eta = vector_transport(X_new, eta)
    PU = X_new.U * X_new.U';
    PV = X_new.V * X_new.V';
    PU_perp = eye(size(PU)) - PU;
    PV_perp = eye(size(PV)) - PV;

    transported_eta = PU * eta * PV_perp + PU_perp * eta * PV + PU * eta * PV;
end

function beta = compute_beta(X_new,alpha, grad, grad_prev, eta_prev, method )
    transported_eta_prev = vector_transport(X_new, eta_prev);
    transported_grad_prev = vector_transport(X_new, grad_prev);
    norm_eta_k = norm(eta_prev, 'fro');
    norm_transported = norm(transported_eta_prev, 'fro');
    scaling_factor = min(1, norm_eta_k / norm_transported);
    eta_scaled = scaling_factor * transported_eta_prev;

    y_k = grad - transported_grad_prev;

    switch method
        case 'HZ'
            numerator = sum(grad(:) .* y_k(:)) ;

            denominator = sum(eta_scaled(:) .* y_k(:)) ;

            mu = 4;
            correction = mu * (norm(y_k, 'fro')^2 / denominator) ...
                            * (sum(grad(:) .* eta_scaled(:)) / norm(eta_scaled, 'fro')^2);

        case 'DY'
            numerator = norm(grad, 'fro')^2;
            term1 = sum(grad(:) .* eta_scaled(:));
            term2 = sum(grad_prev(:) .* eta_prev(:));
            denominator = term1 - term2;
        case 'FR'
            numerator =  norm(grad, 'fro')^2;
            denominator =  norm(grad_prev, 'fro')^2;
        case 'PRP'
            numerator = sum(grad(:) .* y_k(:));
            denominator = norm(grad_prev, 'fro')^2;

        case 'HS'
            numerator = sum(grad(:) .* y_k(:));
            term1 = sum(grad(:) .* eta_scaled(:));
            term2 = sum(grad_prev(:) .* eta_prev(:));
            denominator = term1 - term2;

        case 'NHS'

            grad_k1_norm = norm(grad, 'fro');
            grad_k_norm = norm(grad_prev, 'fro');

            inner_product = sum(grad(:) .* eta_scaled(:));
            numerator =  grad_k1_norm^2  - (grad_k1_norm/grad_k_norm) * abs(inner_product) ;

            denominator = inner_product  - sum(grad_prev(:) .* eta_prev(:)) ;

        case 'Alg1'
            grad_k1_norm = norm(grad, 'fro');
            grad_k_norm = norm(grad_prev, 'fro');

            inner_product = sum(grad(:) .* eta_scaled(:));
            numerator =  grad_k1_norm^2  - (grad_k1_norm/grad_k_norm) * abs(inner_product) ;

            denominator = inner_product  - sum(grad_prev(:) .* eta_prev(:));

        otherwise
            error('未知方法: %s. 可用方法: HZ, FR, DY, PRP, HS, NHS', method);
    end

    if denominator == 0
        beta = 0;

    else

        if strcmp(method, 'HZ')

            beta = (numerator / denominator) - correction;
            beta = max(0, beta);

        else
            beta = numerator / denominator;
        end
    end

    if ismember(upper(method), {'FR', 'DY', 'NHS', 'Alg1'})
        beta = max(0, beta);
    end
end

function M = P_Omega(M, Omega)
    M(~Omega) = 0;
end