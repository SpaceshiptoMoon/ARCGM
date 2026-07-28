function create_lowrank_image(target_rank)
%CREATE_LOWRANK_IMAGE 由原始灰度图经 SVD 截断生成低秩图，供图像修复实验使用。
%
%   create_lowrank_image(target_rank)
%   create_lowrank_image            % 默认 target_rank=60
%
% 输入：repo/data/test_image.png
% 输出：repo/results/rank{N}/rank_{N}_image.{eps,png}、original_grayscale.png

    if nargin < 1
        target_rank = 60;
    end

    % 路径自定位（本文件位于 repo/data_process/image/）
    this_file = mfilename('fullpath');
    this_dir = fileparts(this_file);
    repo_root = fileparts(fileparts(this_dir));
    data_dir = fullfile(repo_root, 'data');
    results_dir = fullfile(repo_root, 'results');

    img = imread(fullfile(data_dir, 'test_image.png'));
    if size(img, 3) == 3
        img = rgb2gray(img);
    end
    img = im2double(img);

    [U, S, V] = svd(img);

    U_trunc = U(:, 1:target_rank);
    S_trunc = S(1:target_rank, 1:target_rank);
    V_trunc = V(:, 1:target_rank);
    low_rank_img = U_trunc * S_trunc * V_trunc';

    rank_dir = fullfile(results_dir, sprintf('rank%d', target_rank));
    if ~exist(rank_dir, 'dir')
        mkdir(rank_dir);
        fprintf('已创建文件夹: %s\n', rank_dir);
    end

    % EPS
    f = figure('Visible', 'off');
    imshow(low_rank_img);
    print(f, fullfile(rank_dir, sprintf('rank_%d_image.eps', target_rank)), '-depsc', '-r300');
    close(f);
    fprintf('EPS图像已保存到: %s\n', fullfile(rank_dir, sprintf('rank_%d_image.eps', target_rank)));

    % 无损 PNG（16-bit）
    imwrite(im2uint16(low_rank_img), ...
        fullfile(rank_dir, sprintf('rank_%d_image.png', target_rank)), 'PNG', 'BitDepth', 16);
    fprintf('无损PNG已保存到: %s\n', fullfile(rank_dir, sprintf('rank_%d_image.png', target_rank)));

    % 原始灰度图（供 svd_analysis.py 估秩对照）
    imwrite(im2uint16(img), fullfile(rank_dir, 'original_grayscale.png'), 'PNG', 'BitDepth', 16);
    fprintf('原始灰度图已保存到: %s\n', fullfile(rank_dir, 'original_grayscale.png'));

    fprintf('\n所有图像格式保存完成！\n');
end