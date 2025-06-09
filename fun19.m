function [error, violation_count,penalty,mse_error, predictions] = fun19(x, inputnum, hiddennum_best, outputnum, input_train, output_train,penalty_weight,input1)
% 提取权重和偏置
w1 = x(1:inputnum * hiddennum_best);
B1 = x(inputnum * hiddennum_best + 1:inputnum * hiddennum_best + hiddennum_best);
w2_start_idx = inputnum * hiddennum_best + hiddennum_best + 1;
w2_end_idx = w2_start_idx + hiddennum_best * outputnum - 1;
w2 = x(w2_start_idx:w2_end_idx);
B2 = x(w2_end_idx + 1:end);

% 重构网络
layers = [
    featureInputLayer(inputnum, 'Normalization', 'none', 'Name', 'input_train')
    fullyConnectedLayer(hiddennum_best, 'Name', 'fc1', 'Weights', reshape(w1, hiddennum_best,inputnum), 'Bias', reshape(B1, hiddennum_best, 1))
    tanhLayer('Name', 'Tanh')  % 使用Tanh作为激活函数
    fullyConnectedLayer(outputnum, 'Name', 'fc2', 'Weights', reshape(w2, outputnum, hiddennum_best), 'Bias', reshape(B2, outputnum, 1))
    reluLayer('Name', 'outputActivation') % 添加ReLU作为输出层的激活函数
    regressionLayer('Name', 'output_train')
    ];

% 重新组装网络
net = assembleNetwork(layers);

% 计算输出和误差
predictions = predict(net, input_train');
mse_error = mean(sqrt((predictions - output_train').^2));
% 惩罚项初始化
penalty = 0;
violation_count=0;
% 对每组输入数据计算激活前输出并检查约束条件
% input_means = mean(input_train, 2);  % 按列计算平均值
input_max2 = max(input1, [], 1);
current_input = input_max2;
% 计算第一个全连接层的激活前输出
pre_activation_output =current_input*reshape(w1, inputnum,hiddennum_best) + B1;
k1= reshape(w1,inputnum,hiddennum_best);
k2=reshape(w2, outputnum, hiddennum_best);
% 检查激活前输出大于0的情况
pos_idx = find(pre_activation_output > 0);
if ~isempty(pos_idx)
    % 如果权值不满足条件（必须为正），增加惩罚项
    if any(k1( :,pos_idx) < 0)
        % 获取符合条件的权值
        k1_pos = k1(:,pos_idx);
        % 通过 logical indexing 筛选出小于 0 的值，并计算其绝对值的和作为惩罚项
        penalty = penalty + sum(abs(k1_pos(k1_pos < 0)));
        % 增加违反约束的神经元计数
        violation_count = violation_count + sum(any(k1( :,pos_idx) < 0));
    end
    % 检查隐藏层到输出层的权重限制
    if any(k2(:,pos_idx) > 0)
        % 获取符合条件的权值
        k2_pos = k2(:, pos_idx);
        % 通过 logical indexing 筛选出大于 0 的权值并计算其绝对值的和作为惩罚项
        penalty = penalty + sum(abs(k2_pos(k2_pos > 0)));
        % 增加违反约束的神经元计数
        violation_count = violation_count + sum(any(k2(:,pos_idx) > 0));
    end
end

% 检查激活前输出小于0的情况
neg_idx = find(pre_activation_output < 0);
if ~isempty(neg_idx)
    % 如果权值不满足条件（必须为负），增加惩罚项
    if any(k1(:,neg_idx) > 0)
        % 获取符合条件的权值
        k1_neg = k1(:,neg_idx);
        % 通过 logical indexing 筛选出大于 0 的值，并计算其绝对值的和作为惩罚项
        penalty = penalty + sum(abs(k1_neg(k1_neg > 0)));
        % 增加违反约束的神经元计数
        violation_count = violation_count + sum(any(k1(:,neg_idx) > 0));
    end

    % 检查隐藏层到输出层的权重限制
    if any(k2(:, neg_idx) < 0)
        % 获取符合条件的权值
        k2_neg = k2(:, neg_idx);
        % 通过 logical indexing 筛选出小于 0 的权值并计算其绝对值的和作为惩罚项
        penalty = penalty + sum(abs(k2_neg(k2_neg < 0)));
        % 增加违反约束的神经元计数
        violation_count = violation_count + sum(any(k2(:, neg_idx) < 0));
    end

end

% 检查输出层偏置是否满足要求（必须为正）
if any(B2 < 0)
    penalty = penalty + sum(abs(B2(B2 < 0)));
    % 增加违反约束的神经元计数
    violation_count = violation_count + sum(any(B2 < 0));
end
% disp(['MSE: ', num2str(mse_error), ', Penalty: ', num2str(penalty), ', Penalty Weight: ', num2str(penalty_weight)]);

% 计算最终适应度值，包含原有的均方误差和惩罚项
error = mse_error +0.01*penalty_weight *penalty; % 惩罚项的权重可根据实际情况调整
end