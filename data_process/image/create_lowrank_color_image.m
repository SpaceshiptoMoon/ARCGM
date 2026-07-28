function create_lowrank_color_image(target_rank)
%CREATE_LOWRANK_COLOR_IMAGE 由原始彩色图逐通道 SVD 截断生成低秩彩色图。
%
%   create_lowrank_color_image(target_rank)
%   create_lowrank_color_image      % 默认 target_rank=80
%
% 输入：repo/data/test_color_image.png
% 输出：repo/results/color/rank{N}/rank_{N}_image.png

    if nargin < 1
        target_rank = 80;
    end

    % 路径自定位（本文件位于 repo/data_process/image/）
    this_file = mfilename('fullpath');
    this_dir = fileparts(this_file);
    repo_root = fileparts(fileparts(this_dir));
    data_dir = fullfile(repo_root, 'data');
    results_dir = fullfile(repo_root, 'results');

    img = imread(fullfile(data_dir, 'test_color_image.png'));
    img = im2double(img);

    low_rank_img = zeros(size(img));
    for channel = 1:3
        [U, S, V] = svd(img(:,:,channel));
        U_trunc = U(:, 1:target_rank);
        S_trunc = S(1:target_rank, 1:target_rank);
        V_trunc = V(:, 1:target_rank);
        low_rank_img(:,:,channel) = U_trunc * S_trunc * V_trunc';
    end

    rank_dir = fullfile(results_dir, 'color', sprintf('rank%d', target_rank));
    if ~exist(rank_dir, 'dir')
        mkdir(rank_dir);
        fprintf('已创建文件夹: %s\n', rank_dir);
    end

    imwrite(low_rank_img, fullfile(rank_dir, sprintf('rank_%d_image.png', target_rank)));
    fprintf('低秩彩色图像已保存到: %s\n', fullfile(rank_dir, sprintf('rank_%d_image.png', target_rank)));
end