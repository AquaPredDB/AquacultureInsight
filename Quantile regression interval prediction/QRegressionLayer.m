classdef QRegressionLayer < nnet.layer.RegressionLayer
    
    % 自定义回归层
    properties
        tau
    end

    methods

        function layer = QRegressionLayer(name, tau)
            
            % 创建回归层，输入名称和参数
            layer.tau = tau;                       % 设置网络层参数
            layer.Name = name;                     % 设置网络层名称
            layer.Description = 'quantile error';  % 设置网络层的描述

        end
        
        function loss = forwardLoss(layer, Y, T)
            % 得到真实值T和预测值Y
            % 计算平均绝对误差
            R = size(Y, 1);
            quantileError = sum(max(layer.tau * (T - Y), (1 - layer.tau) * (Y - T))) / R;  % 有部分点不可导，可以改进
            N = size(Y, 3);
            loss = sum(quantileError) / N;
        end

        % 损失函数梯度下降计算层
        function dLdY = backwardLoss(layer, Y, T)
           
            dLdY =  single(-layer.tau * (T - Y >= 0) + (1 - layer.tau) * (Y - T >= 0));
                    
        end

    end

end