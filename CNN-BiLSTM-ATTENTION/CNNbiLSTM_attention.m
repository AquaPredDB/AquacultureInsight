%%  清空环境变量
warning off             % 关闭报警信息
close all               % 关闭开启的图窗
clear                   % 清空变量
clc  

%% 导入数据
data =  readmatrix('../day.csv');
data = data(:,3:16);
res=data(randperm(size(data,1)),:);    %此行代码用于打乱原始样本，使训练集测试集随机被抽取，有助于更新预测结果。
num_samples = size(res,1);   %样本个数


% 训练集和测试集划分
outdim = 1;                                  % 最后一列为输出
num_size = 0.7;                              % 训练集占数据集比例
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

%%  数据平铺

for i = 1:size(P_train,2)
    trainD{i,:} = (reshape(p_train(:,i),size(p_train,1),1,1));
end

for i = 1:size(p_test,2)
    testD{i,:} = (reshape(p_test(:,i),size(p_test,1),1,1));
end


targetD =  t_train;
targetD_test  =  t_test;

numFeatures = size(p_train,1);


layers0 = [ ...
    % 输入特征
    sequenceInputLayer([numFeatures,1,1],'name','input')   %输入层设置
    sequenceFoldingLayer('name','fold')         %使用序列折叠层对图像序列的时间步长进行独立的卷积运算。
    % CNN特征提取
    convolution2dLayer([3,1],16,'Stride',[1,1],'name','conv1')  %添加卷积层，64，1表示过滤器大小，10过滤器个数，Stride是垂直和水平过滤的步长
    batchNormalizationLayer('name','batchnorm1')  % BN层，用于加速训练过程，防止梯度消失或梯度爆炸
    reluLayer('name','relu1')       % ReLU激活层，用于保持输出的非线性性及修正梯度的问题
      % 池化层
    maxPooling2dLayer([2,1],'Stride',2,'Padding','same','name','maxpool')   % 第一层池化层，包括3x3大小的池化窗口，步长为1，same填充方式
    % 展开层
    sequenceUnfoldingLayer('name','unfold')       %独立的卷积运行结束后，要将序列恢复
    %平滑层
    flattenLayer('name','flatten')
    
    bilstmLayer(25,'Outputmode','last','name','hidden1') 
    selfAttentionLayer(1,2)          %创建一个单头，2个键和查询通道的自注意力层  
    dropoutLayer(0.1,'name','dropout_1')        % Dropout层，以概率为0.2丢弃输入

    fullyConnectedLayer(1,'name','fullconnect')   % 全连接层设置（影响输出维度）（cell层出来的输出层） %
    regressionLayer('Name','output')    ];
    
lgraph0 = layerGraph(layers0);
lgraph0 = connectLayers(lgraph0,'fold/miniBatchSize','unfold/miniBatchSize');


%% Set the hyper parameters for unet training
options0 = trainingOptions('adam', ...                 % 优化算法Adam
    'MaxEpochs', 150, ...                            % 最大训练次数
    'GradientThreshold', 1, ...                       % 梯度阈值
    'InitialLearnRate', 0.01, ...         % 初始学习率
    'LearnRateSchedule', 'piecewise', ...             % 学习率调整
    'LearnRateDropPeriod',100, ...                   % 训练100次后开始调整学习率
    'LearnRateDropFactor',0.01, ...                    % 学习率调整因子
    'L2Regularization', 0.001, ...         % 正则化参数
    'ExecutionEnvironment', 'cpu',...                 % 训练环境
    'Verbose', 1, ...                                 % 关闭优化过程
    'Plots', 'none');                    % 画出曲线
% % start training
%  训练
tic
net = trainNetwork(trainD,targetD',lgraph0,options0);
toc
%analyzeNetwork(net);% 查看网络结构
%  预测
t_sim1 = predict(net, trainD); 
t_sim2 = predict(net, testD); 

%  数据反归一化
T_sim1 = mapminmax('reverse', t_sim1, ps_output);
T_sim2 = mapminmax('reverse', t_sim2, ps_output);
T_train1 = T_train;
T_test2 = T_test;

%  数据格式转换
T_sim1 = double(T_sim1);% cell2mat将cell元胞数组转换为普通数组
T_sim2 = double(T_sim2);

CNNBILSTM_attention_TSIM1 = T_sim1';
CNNBILSTM_attention_TSIM2 = T_sim2';
save CNNBILSTM_attention CNNBILSTM_attention_TSIM1 CNNBILSTM_attention_TSIM2



% 指标计算
disp('…………训练集误差指标…………')
[mae1,rmse1,mape1,error1]=calc_error(T_train1,T_sim1');
fprintf('\n')

figure('Position',[200,300,1200,250])
plot(T_train1,'b-o');
hold on
plot(T_sim1','r-*');
legend('真实值','回归拟合值')
title('CNN-BiLSTM-Attention训练集回归拟合效果对比')
xlabel('样本点')
ylabel('自行车租赁数量')

disp('…………CNN-BiLSTM-Attention测试集误差指标…………')
[mae2,rmse2,mape2,error2]=calc_error(T_test2,T_sim2');
fprintf('\n')


figure('Position',[200,300,900,250])
plot(T_test2,'b-o');
hold on
plot(T_sim2','r-*');
legend('真实值','回归拟合值')
title('CNN-BiLSTM-Attention测试集回归拟合效果对比')
xlabel('样本点')
ylabel('自行车租赁数量')

figure('Position',[200,300,900,250])
plot(T_sim2'-T_test2)
title('CNN-BiLSTM-Attention误差曲线图')
xlabel('样本点')
ylabel('自行车租赁数量')