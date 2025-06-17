
%% 此程序为单变量输入单步预测
%%  清空环境变量
warning off             % 关闭报警信息
close all               % 关闭开启的图窗
clear                   % 清空变量
clc  

%% 导入数据
data =  readmatrix('风电场预测.xlsx');
data = data(5665:6665,15);  %选取部分数据，第15列为风电功率
[h1,l1]=data_process(data,8);   %单步预测%步长为8，采用前8个时刻的风电功率预测第9个时刻的风电功率
res = [h1,l1];
num_samples = size(res,1);   %样本个数
% 训练集和测试集划分
outdim = 1;                                  % 最后一列为输出
num_size = 0.8;                              % 训练集占数据集比例
num_train_s = round(num_size * num_samples); % 训练集样本个数
f_ = size(res, 2) - outdim;                  % 输入特征维度


P_train = res(1: num_train_s, 1: f_)';
T_train = res(1: num_train_s, f_ + 1: end)';
M = size(P_train, 2);

P_test = res(num_train_s + 1: end, 1: f_)';
T_test = res(num_train_s + 1: end, f_ + 1: end)';
N = size(P_test, 2);

%  数据归一化
[p_train, ps_input] = mapminmax(P_train, 0, 1);
p_test = mapminmax('apply', P_test, ps_input);

[t_train, ps_output] = mapminmax(T_train, 0, 1);
t_test = mapminmax('apply', T_test, ps_output);

%  格式转换
for i = 1 : M 
    vp_train{i, 1} = p_train(:, i);
    vt_train{i, 1} = t_train(:, i);
end

for i = 1 : N 
    vp_test{i, 1} = p_test(:, i);
    vt_test{i, 1} = t_test(:, i);
end

save_net = [];
for i = 0.02 : 0.05 : 0.97                                      % 置信区间范围 0.97 - 0.02 = 0.95
%% 网络搭建
numFeatures = size(p_train,1);

%% 网络搭建CNN-BiGRU-Attention
lgraph = layerGraph();

% 添加层分支
% 将网络分支添加到层次图中。每个分支均为一个线性层组。
tempLayers = sequenceInputLayer([numFeatures,1,1],"Name","sequence");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([2 1],8,"Name","conv","Padding","same")
    batchNormalizationLayer("Name","batchnorm")
    reluLayer("Name","relu")
    maxPooling2dLayer([2 1],"Name","maxpool","Padding","same")
    flattenLayer("Name","flatten_1")
    fullyConnectedLayer(10,"Name","fc_1")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = flattenLayer("Name","flatten");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = gruLayer(10,"Name","gru1");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    FlipLayer("flip3")
    gruLayer(10,"Name","gru2")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    concatenationLayer(1,3,"Name","concat")
    fullyConnectedLayer(outdim,"Name","fc")
    selfAttentionLayer(1,5,"Name","selfattention")   %Attention机制
    QRegressionLayer('out', i)];

lgraph = addLayers(lgraph,tempLayers);

% 清理辅助变量
clear tempLayers;

% 连接层分支
% 连接网络的所有分支以创建网络图。
lgraph = connectLayers(lgraph,"sequence","conv");
lgraph = connectLayers(lgraph,"sequence","flatten");
lgraph = connectLayers(lgraph,"flatten","gru1");
lgraph = connectLayers(lgraph,"flatten","flip3");
lgraph = connectLayers(lgraph,"gru1","concat/in1");
lgraph = connectLayers(lgraph,"gru2","concat/in2");
lgraph = connectLayers(lgraph,"fc_1","concat/in3");


%  参数设置
options = trainingOptions('adam', ...                 % 优化算法Adam
    'MaxEpochs', 150, ...                            % 最大训练次数
    'GradientThreshold', 1, ...                       % 梯度阈值
    'InitialLearnRate', 0.001, ...         % 初始学习率
    'Shuffle', 'every-epoch', ...          % 训练打乱数据集
    'ExecutionEnvironment', 'cpu',...                 % 训练环境
    'Verbose', 1, ...                                 % 关闭优化过程
    'Plots', 'none');                    % 画出曲线

%  训练

net = trainNetwork(vp_train, vt_train, lgraph, options);
   

    % 保存网络
    save_net = [save_net, net];

end

%%  采用不同网络进行预测
for i = 1 : length(save_net)
    i
    % 仿真预测
    t_sim1(i, :) = predict(save_net(i), vp_train); 
    t_sim2(i, :) = predict(save_net(i), vp_test ); 

    % 数据反归一化
    L_sim1{i} = cell2mat(mapminmax('reverse', t_sim1(i, :), ps_output));
    L_sim2{i} = cell2mat(mapminmax('reverse', t_sim2(i, :), ps_output));

    tt_sim1(i, :) = cell2mat(mapminmax('reverse', t_sim1(i, :), ps_output));
    tt_sim2(i, :) = cell2mat(mapminmax('reverse', t_sim2(i, :), ps_output));

end


%%  得到预测均值
T_sim1 = mean(tt_sim1); 
T_sim2 = mean(tt_sim2); 

%%  性能评估
error1 = sqrt(sum((T_sim1 - T_train) .^2 ) ./ M);
error2 = sqrt(sum((T_sim2 - T_test ) .^2 ) ./ N);

%%  绘图
figure
fill([1 : M, M : -1 : 1], [L_sim1{1}, L_sim1{end}(end : -1 : 1)], ...
    'r', 'FaceColor', [1, 0.8, 0.8], 'EdgeColor', 'none')
hold on 
plot(1 : M, T_train, 'r-', 1 : M, T_sim1', 'b-', 'LineWidth', 0.3)
legend('95%的置信区间', '真实值', '预测值')
xlabel('预测样本')
ylabel('预测结果')
string = {'QRCNN-BiGRU-Attention训练集预测结果对比'; ['RMSE = ' num2str(error1)]};
title(string)
xlim([1, M])
grid

figure
fill([1 : N, N : -1 : 1], [L_sim2{1}, L_sim2{end}(end : -1 : 1)], ...
    'r', 'FaceColor', [1, 0.8, 0.8], 'EdgeColor', 'none')
hold on 
plot(1 : N, T_test, 'r-', 1 : N, T_sim2', 'b-', 'LineWidth', 1)
legend('95%的置信区间', '真实值', '预测值')
xlabel('预测样本')
ylabel('预测结果')
string = {'QRCNN-BiGRU-Attention测试集预测结果对比'; ['RMSE = ' num2str(error2)]};
title(string)
xlim([1, N])
grid

%%  相关指标计算
% 指标计算
disp('…………QRCNN-BiGRU-Attention训练集误差指标…………')
[mae1,rmse1,mape1,error1]=calc_error(T_train,T_sim1);
fprintf('\n')

disp('…………QRCNN-BiGRU-Attention测试集误差指标…………')
[mae1,rmse1,mape1,error1]=calc_error(T_test,T_sim2);
fprintf('\n')

%%  指标计算(区间覆盖率和区间平均宽度百分比)

picp1 = PICP (tt_sim1, T_train');
pinaw1 = PINAW(tt_sim1, T_train');
disp(['训练集的区间覆盖率为:', num2str(picp1), '。区间平均宽度百分比为:', num2str(pinaw1)])

picp2 = PICP (tt_sim2, T_test');
pinaw2 = PINAW(tt_sim2, T_test');
disp(['测试集的区间覆盖率为:', num2str(picp2), '。区间平均宽度百分比为:', num2str(pinaw2)])

