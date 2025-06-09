classdef CustomRegressionLayer19 < nnet.layer.RegressionLayer
    % 自定义回归层以添加物理信息的约束
    properties
        % 可以定义额外的属性
        InputData
        OutputData
    end

    methods
        function layer = CustomRegressionLayer19(name, inputData, outputData)
            % 创建一个自定义回归层
            layer.Name = name;

            % 设置层描述
            layer.Description = 'Custom Regression with physical constraints';

            % 存储输入数据
            layer.InputData = inputData;
            layer.OutputData = outputData;

        end

        function loss = forwardLoss(layer, Y, T)
            % 前向传播定义损失函数
            % Y为预测值，T为真实值

            % 确保Y和T都是dlarray对象
            if ~isa(Y, 'dlarray')
                Y = dlarray(Y);
            end

            if ~isa(T, 'dlarray')
                T = dlarray(T);
            end

            % 计算标准均方误差
            meanSquaredError = mean(sqrt(Y - T).^2, 'all');

            % 读取归一化前输入数据的相关列

            n = dlarray(layer.InputData(:, 6));        % 第4列对应n
            Nlcf = dlarray(layer.InputData(:, 7));     % 第5列对应N_{LCF}
            Nhcf = dlarray(layer.InputData(:, 8));     % 第6列对应N_{HCF}

            % 计算物理约束

            % 计算物理约束损失
            Nf = (1 + n) ./ ((1./Nlcf)+n./ Nhcf);

            % 对输出数据 output1 进行标准化
            
            % 获取输出特征的最小值和最大值
            output_min = min( layer.OutputData, [], 1);  % 每列的最小值
            output_max = max( layer.OutputData, [], 1);  % 每列的最大值

            % 防止特征所有值相同，出现除以零的问题
            output_range = output_max - output_min;
            output_range(output_range == 0) = 1;  % 如果 range 为 0，设置为 1

            % 对输出特征进行归一化
            Nf1 = (Nf - output_min) ./ output_range;

            physicalLoss = mean(( Nf1-T ).^2, 'all');

            % 总损失是MSE和物理损失的组合
            loss = meanSquaredError+2*physicalLoss;
        end
    end
end