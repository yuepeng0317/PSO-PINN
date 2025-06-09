%% 清空环境
clc;
clear;
%% 读取数据
load data16(1).mat
% 获取样本数量.
[m, ~] = size(input);

% 生成一个随机排列的索引向量
% 生成一个随机排列的索引向量
% randomOrder = [4 6 1 3 2 5];
randomOrder = randperm(m);
% 根据随机排列的索引向量重新排列特征矩阵和标签向量
input1 = input(randomOrder, :);
output1 = output(randomOrder, :);
input_train1=input1(1:5, :)';
input1(:, [2,6,7]) = [];
%归一化
%输入归一化
% === 输入特征的最小-最大归一化 ===
% 获取输入特征的最小值和最大值
input_min = min(input1, [], 1);  % 每列的最小值
input_max1 = max(input1, [], 1);  % 每列的最大值

% 防止特征所有值相同，出现除以零的问题
input_range = input_max1 - input_min;
input_range(input_range == 0) = 1;  % 如果 range 为 0，设置为 1

% 对输入特征进行归一化
inputn1 = (input1 - input_min) ./ input_range;
% 输出归一化
% === 输出特征的最小-最大归一化 ===
% 获取输出特征的最小值和最大值

output_min = min(output1, [], 1);  % 每列的最小值
output_max = max(output1, [], 1);  % 每列的最大值

% 防止特征所有值相同，出现除以零的问题
output_range = output_max - output_min;
output_range(output_range == 0) = 1;  % 如果 range 为 0，设置为 1

% 对输出特征进行归一化
outputn1 = (output1 - output_min) ./ output_range;

% 节点个数
inputnum = size(input1, 2); % 输入层神经元节点个数
outputnum = size(output1, 2); % 输出层神经元节点个数
input_train = inputn1(1:5, :)';
input_test = inputn1(6:end, :)';
output_train = outputn1(1:5)';
output_test = outputn1(6:end)';

input_max = max(input_train1, [], 2);
%隐藏层神经元寻优
% 采用经验公式 hiddennum = sqrt(m + n) + a，m 为输入层节点个数，n 为输出层节点个数，a 一般取为 1-10 之间的整数
disp(['输入层节点数：', num2str(inputnum), ',  输出层节点数：', num2str(outputnum)])
disp(['隐含层节点数范围为 ', num2str(fix(sqrt(inputnum + outputnum)) + 3), ' 至 ', num2str(fix(sqrt(inputnum + outputnum)) + 10)])
disp(' ')
disp('最佳隐含层节点的确定...')

MSE = 1e+14;  % 误差初始化
transform_func = {'tansig', 'purelin'};  % 激活函数采用 tan-sigmoid 和 purelin
train_func = 'trainlm';  % 训练算法
for hiddennum = fix(sqrt(inputnum + outputnum)) + 3 : fix(sqrt(inputnum + outputnum)) + 13
    net1 = newff(input_train, output_train, hiddennum, transform_func, train_func);  % 构建 BP 网络

    % 设置网络参数
    net1.trainParam.epochs = 1000;  % 设置训练次数
    net1.trainParam.lr = 0.01;  % 设置学习速率
    net1.trainParam.goal = 0.000001;  % 设置训练目标最小误差

    % 进行网络训练
    net1 = train(net1, input_train, output_train);
    T_test_predict = sim(net1, input_train);  % 仿真结果
    mse0 = mse(output_train, T_test_predict);  % 仿真的均方误差
    disp(['当隐含层节点数为', num2str(hiddennum), '时，训练集均方误差为：', num2str(mse0)])

    % 不断更新最佳隐含层节点
    if mse0 < MSE
        MSE = mse0;
        hiddennum_best = hiddennum;
    end
end
disp(['最佳隐含层节点数为：', num2str(hiddennum_best), '，均方误差为：', num2str(MSE)])
hiddennum =hiddennum_best;
%% 自定义神经网络
layers = [
    featureInputLayer(inputnum, 'Normalization', 'none', 'Name', 'input_train')
    fullyConnectedLayer(hiddennum_best, 'Name', 'fc1')
    tanhLayer('Name', 'Tanh')  % 使用Tanh作为激活函数
    fullyConnectedLayer(outputnum, 'Name', 'fc2')
    reluLayer('Name', 'outputActivation') % 添加ReLU作为输出层的激活函数
    CustomRegressionLayer19('customOutput',input_train1',output1)
];

% 选项设置
options = trainingOptions('adam', ...
    'MaxEpochs', 500, ...
    'MiniBatchSize', 5, ...
    'InitialLearnRate', 0.0001, ...
    'ValidationData', {input_test', output_test'}, ...
    'ValidationFrequency', 10, ...
    'Plots', 'none');
% 训练网络
net = trainNetwork(input_train', output_train', layers, options);

%% 粒子群优化
% 粒子群算法中的两个参数
c1 = 0.2;
c2 = 0.2;
maxgen = 40;  % 进化次数
sizepop = 30;  % 种群规模
Vmax = 0.6;
Vmin = -0.6;
popmax =1;
popmin = -1;
% 初始化惩罚项权重
penalty_weight_initial = 1;  % 初始的惩罚权重
penalty_weight = 1;
penalty_increase_factor = 8.5;  % 惩罚项增加因子
penalty_decrease_factor = 1.5;  % 惩罚项减少因子
best_gen = 1;  % 最佳的代数
best_particle = 1;  % 最佳的粒子编号
a = inputnum * hiddennum_best + hiddennum_best + hiddennum_best * outputnum + outputnum;
for i = 1:sizepop
    % 隐藏层权值和偏置初始化
    w1_range = sqrt(6 / (inputnum + hiddennum_best));  % Xavier 初始化范围
    w1 = (rand(inputnum, hiddennum_best) - 0.5) * 2 * w1_range;  % Xavier 初始化
    b1 = (rand(hiddennum_best, 1) - 0.5) * 2 * w1_range;  % 偏置用相同策略初始化

    % ------------------ 输出层权值和偏置初始化 (ReLU - He) -------------------
    w2_std = sqrt(2 / hiddennum_best);  % He 初始化标准差
    w2 = normrnd(0, w2_std, outputnum, hiddennum_best);  % He 初始化
    b2 = zeros(outputnum, 1);  % 偏置初始化为零，ReLU 推荐这样处理

    % 计算激活前输出并调整权值

    current_input = input_max1;  % 第 n 组输入数据

    % 计算第 n 组输入数据的激活前输出
    pre_activation_output = current_input*w1  + b1';

    % 对于 pre_activation_output > 0 的神经元，确保输入权值 > 0，输出权值 < 0
    pos_idx = pre_activation_output > 0;
    w1(:,pos_idx) = abs(w1(:,pos_idx));  % 确保输入权值为正
    w2(:, pos_idx) = -abs(w2(:, pos_idx));  % 确保输出权值为负

    % 对于 pre_activation_output < 0 的神经元，确保输入权值 < 0，输出权值 > 0
    neg_idx = pre_activation_output < 0;
    w1(:,neg_idx) = -abs(w1(:,neg_idx));  % 确保输入权值为负
    w2(:, neg_idx) = abs(w2(:, neg_idx));  % 确保输出权值为正


    % 将权值和偏置合并成粒子个体
    % **将所有参数按行转换为一维向量**
    w1_flat = reshape(w1, 1, []);  % 隐藏层权值按行展平为一维向量
    b1_flat = reshape(b1, 1, []);  % 隐藏层偏置按行展平为一维向量
    w2_flat = reshape(w2, 1, []);  % 输出层权值按行展平为一维向量
    b2_flat = reshape(b2, 1, []);  % 输出层偏置按行展平为一维向量

    % 将所有参数合并为一个一维向量
    pop(i, :) = [w1_flat b1_flat w2_flat b2_flat];  % 按顺序合并为一维向量

    % 初始化速度
    V(i, :) = normrnd(0, 0.01, 1, numel(pop(i, :)));  % 速度初始化为小随机值

    % 计算适应度
    fitness(i) = fun19(pop(i, :), inputnum, hiddennum_best, outputnum, input_train, output_train,penalty_weight,input1);
end

% 个体极值和群体极值
[bestfitness, bestindex] = min(fitness);
zbest = pop(bestindex, :);  % 全局最佳
gbest = pop;  % 个体最佳
fitnessgbest = fitness;  % 个体最佳适应度值
fitnesszbest = bestfitness;  % 全局最佳适应度值
% 迭代寻优
for gen = 1:maxgen
    for j = 1:sizepop
        % 速度更新
        V(j, :) = V(j, :) + c1 * rand * (gbest(j, :) - pop(j, :)) + c2 * rand * (zbest - pop(j, :));
        V(j, find(V(j, :) > Vmax)) = Vmax;
        V(j, find(V(j, :) < Vmin)) = Vmin;

        % 种群更新
        pop(j, :) = pop(j, :) + 0.2 * V(j, :);
        pop(j, find(pop(j, :) > popmax)) = popmax;
        pop(j, find(pop(j, :) < popmin)) = popmin;

        % 自适应变异
        pos = unidrnd(21);
        if rand > 0.7
            pop(j, pos) = 5 * rands(1, 1);
        end

        % 适应度值
        [fitness(j), violation_count(j),penalty(j),best_mse_error] = fun19(pop(j, :), inputnum, hiddennum_best, outputnum,input_train, output_train,penalty_weight,input1);
    end

    for j = 1:sizepop
        % 个体最优更新
        if fitness(j) < fitnessgbest(j)
            gbest(j, :) = pop(j, :);
            fitnessgbest(j) = fitness(j);
        end

        % 群体最优更新
        if fitness(j) < fitnesszbest
            zbest = pop(j, :);
            fitnesszbest = fitness(j);
            best_gen = gen;  % 记录最佳的代数
            best_particle = j;  % 记录最佳的粒子编号
            % 调用适应度函数fun10时，保存返回的mse_error和penalty等
            [best_error, ~, best_penalty, best_mse_error, predictions] = fun19(pop(j, :), inputnum, hiddennum_best, outputnum, input_train, output_train, penalty_weight,input1);
            best_predictions = predictions;  % 保存最佳粒子的预测值
            best_penalty_weight = penalty_weight;  % 保存当时的惩罚权重
        end
        % 增加约束检查和修正 
        % 检查是否满足物理约束，并进行修正
        % 提取更新后的权重和偏置
        w1 = reshape(pop(j, 1:inputnum * hiddennum_best), inputnum,hiddennum_best );
        b1 = pop(j, inputnum * hiddennum_best + 1:inputnum * hiddennum_best + hiddennum_best);
        w2_start_idx = inputnum * hiddennum_best + hiddennum_best + 1;
        w2_end_idx = w2_start_idx + hiddennum_best * outputnum - 1;
        w2 = reshape(pop(j, w2_start_idx:w2_end_idx), outputnum, hiddennum_best);
        b2 = pop(j, w2_end_idx + 1:end);

        % 检查是否违反约束
        current_input = input_max1;  % 第 n 组输入数据
        pre_activation_output = current_input*w1  + b1;

        % 修正 pre_activation_output > 0 的神经元，确保输入权值 > 0，输出权值 < 0
        pos_idx = pre_activation_output > 0;
        w1(:,pos_idx) = abs( w1(:,pos_idx));  % 输入层权值修正为正
        w2(:, pos_idx) = -abs(w2(:, pos_idx));  % 隐藏层到输出层的权值修正为负

        % 修正 pre_activation_output < 0 的神经元，确保输入权值 < 0，输出权值 > 0
        neg_idx = pre_activation_output < 0;
        w1(:,neg_idx) = -abs(w1(:,neg_idx));  % 输入层权值修正为负
        w2(:, neg_idx) = abs(w2(:, neg_idx));  % 隐藏层到输出层的权值修正为正

        % **将所有参数按行转换为一维向量**
        w1_flat2 = reshape(w1, 1, []);  % 隐藏层权值按列展平为一维向量
        b1_flat2 = reshape(b1, 1, []);  % 隐藏层偏置按列展平为一维向量
        w2_flat2 = reshape(w2, 1, []);  % 输出层权值按列展平为一维向量
        b2_flat2 = reshape(b2, 1, []);  % 输出层偏置按列展平为一维向量
        % 将修正后的权值和偏置重新展平为一维向量，并存回 pop(j, :)
        pop(j, :) =[w1_flat2 b1_flat2 w2_flat2 b2_flat2];  % 按顺序合并为一维向量
    end
    %计算违反约束的粒子比例
    total_violations = sum(violation_count);  % 计算所有违反约束的神经元个数
    violation_ratio = total_violations / (sizepop * (hiddennum_best+outputnum));  % 计算违反约束的比例

    %动态调整惩罚项权重
    if violation_ratio > 0.2  % 如果超过80%的神经元违反了约束条件
        penalty_weight = penalty_weight * penalty_increase_factor;  % 增加惩罚权重
    else
        penalty_weight=penalty_weight_initial;
        penalty_weight = penalty_weight * penalty_decrease_factor;  % 减少惩罚权重
    end
    % 存储每代的适应度和违反约束情况
    violations_history(gen) = total_violations;  % 每一代的违反约束神经元个数
    yy(gen) = fitnesszbest;
    penalty_weight_history(gen)=penalty_weight;
    % 在本次循环结束时重置penalty_weight
    if  mod(gen, 2) == 0    % 每隔一次迭代重置
        penalty_weight = penalty_weight_initial;
    end
end
% 结果分析
plot(yy)
title(['适应度曲线  ' '终止代数＝' num2str(maxgen)]);
xlabel('进化代数');
ylabel('适应度');
% 绘制违反约束神经元个数的变化曲线
figure;
plot(violations_history);
title('违反约束的神经元个数随代数的变化');
xlabel('进化代数');
ylabel('违反约束的神经元个数');
X = zbest;
disp(['最佳的代数是: ', num2str(best_gen)]);
disp(['最佳的粒子编号是: ', num2str(best_particle)]);
disp(['最佳粒子的适应度是: ', num2str(fitnesszbest)]);
disp(['最佳粒子的MSE误差是: ', num2str(best_mse_error)]);
disp(['最佳粒子的惩罚项是: ', num2str(best_penalty)]);
disp(['最佳惩罚权重是: ', num2str(best_penalty_weight)]);
disp(['最佳粒子的参数是: ']);
disp(zbest);
disp('最佳粒子的预测值为:');
disp(best_predictions);
% 提取权重和偏置
w1 = X(1:inputnum * hiddennum_best);
B1 = X(inputnum * hiddennum_best + 1:inputnum * hiddennum_best + hiddennum_best);
w2_start_idx = inputnum * hiddennum_best + hiddennum_best + 1;
w2_end_idx = w2_start_idx + hiddennum_best * outputnum - 1;
w2 = X(w2_start_idx:w2_end_idx);
B2 = X(w2_end_idx + 1:end);

% 重构网络
layers = [
    featureInputLayer(inputnum, 'Normalization', 'none', 'Name', 'input_train')
    fullyConnectedLayer(hiddennum_best, 'Name', 'fc1', 'Weights', reshape(w1, hiddennum_best,inputnum), 'Bias', reshape(B1, hiddennum_best, 1))
    tanhLayer('Name', 'Tanh')  % 使用Tanh作为激活函数
    fullyConnectedLayer(outputnum, 'Name', 'fc2', 'Weights', reshape(w2, outputnum, hiddennum_best), 'Bias', reshape(B2, outputnum, 1))
    reluLayer('Name', 'outputActivation') % 添加ReLU作为输出层的激活函数
    CustomRegressionLayer19('customOutput',input_train1',output)
    ];

% 重新组装网络
net = assembleNetwork(layers);


%% 测试集预测
Y_pred = predict(net, inputn1);
% 对输出数据进行反归一化
Y_pred = Y_pred .* output_range + output_min;
actual_pred = Y_pred';
actual_output=output1';
relative_error=(actual_pred-actual_output)./actual_output;
% 将相对误差转换为百分比
relative_error_percent = relative_error * 100;
% 打印结果
fprintf('Relative Error: %.2f%%\n', relative_error_percent);

% 绘图
figure;
plot(actual_output, 'b:*');
hold on;
plot(actual_pred, 'r-o');
legend('实际值', '预测值');
xlabel('样本索引');
ylabel('值');
title('测试集实际值与预测值对比');
hold off;

%% 精度分析
experimental_life = actual_output; % 实验寿命数据
predicted_life = actual_pred; % 预测寿命数据

% 找到预测值的最小和最大范围，并略微扩展范围
min_life = min([experimental_life, predicted_life]) * 0.9;
max_life = max([experimental_life, predicted_life]) * 1.1;

% 绘制实际寿命与预测寿命的关系
figure;
h0 = loglog(experimental_life, predicted_life, 'k*', 'MarkerSize', 10); % 绘制实验寿命与预测寿命的数据点

hold on;
% 绘制从最小值到最大值的45度参考线
h1 = loglog([min_life max_life], [min_life max_life], 'k-', 'LineWidth', 1.5);

% 绘制±1.5倍误差带
h2 = loglog([min_life max_life], 2*[min_life max_life], 'b-.', 'LineWidth', 1.5);
loglog([min_life max_life], [min_life max_life] / 2, 'b-.', 'LineWidth', 1.5);

% 绘制±2倍误差带，设置为红色
h3 = loglog([min_life max_life], 3*[min_life max_life], 'r--', 'LineWidth', 1.5);
loglog([min_life max_life], [min_life max_life] / 3, 'r--', 'LineWidth', 1.5);

% 图形美化
xlabel('Experimental life');
ylabel('Predicted life');
title('TC4');
legend([h0 h2 h3], 'Training data',  '\pm2 life factor', '\pm3 life factor', 'Location', 'Northwest');
grid on;
hold off;
% 设置轴的范围
xlim([min_life max_life]);
ylim([min_life max_life]);

% 自动生成刻度
min_log10 = floor(log10(min_life));
max_log10 = ceil(log10(max_life));
xticks = 10.^(min_log10:max_log10);
yticks = 10.^(min_log10:max_log10);
set(gca, 'XTick', xticks);
set(gca, 'YTick', yticks);

% 自动生成刻度标签
xtick_labels = arrayfun(@(x) sprintf('10^{%d}', log10(x)), xticks, 'UniformOutput', false);
ytick_labels = arrayfun(@(x) sprintf('10^{%d}', log10(x)), yticks, 'UniformOutput', false);
set(gca, 'XTickLabel', xtick_labels);
set(gca, 'YTickLabel', ytick_labels);

% 强制刷新图形以显示新标签
drawnow;