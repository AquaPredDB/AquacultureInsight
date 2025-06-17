%%  清空环境变量
warning off             % 关闭报警信息
close all               % 关闭开启的图窗
clear                   % 清空变量
clc  

%% 导入数据
data =  readmatrix('../风电场预测.xlsx');
data = data(5665:8640,12);  %选取3月份数据,第12列为温度数据，单变量的意思是只选取这一列的变量
nn =8;   %预测未来八个时刻的数据
[h1,l1]=data_process(data,24,nn);   %步长为24，采用前24个时刻的温度预测第25~24+nn个时刻的温度
res = [h1,l1];
num_samples = size(res,1);   %样本个数


% 训练集和测试集划分
outdim = nn;                                  % 最后nn列为输出
num_train_s = num_samples-1; % 训练集样本个数
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

    fullyConnectedLayer(outdim,'name','fullconnect')   % 全连接层设置（影响输出维度）（cell层出来的输出层） %
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

t_sim= predict(net, testD); 

%  数据反归一化

T_sim = mapminmax('reverse', t_sim', ps_output);



%% 比较算法预测值
str={'真实值','CNN-BiLSTM-Attention'};
figure('Units', 'pixels', ...
    'Position', [300 300 860 370]);
plot(T_test,'--*') 
hold on
plot(T_sim,'-.p')
legend(str)
set (gca,"FontSize",12,'LineWidth',1.2)
box off
legend Box off



%% 比较算法误差
test_y = T_test;
Test_all = [];

y_test_predict = T_sim;
[test_MAE,test_MAPE,test_MSE,test_RMSE,test_R2]=calc_error(y_test_predict,test_y);


Test_all=[Test_all;test_MAE test_MAPE test_MSE test_RMSE test_R2];



str={'真实值','CNN-BiLSTM-Attention'};
str1=str(2:end);
str2={'MAE','MAPE','MSE','RMSE','R2'};
data_out=array2table(Test_all);
data_out.Properties.VariableNames=str2;
data_out.Properties.RowNames=str1;
disp(data_out)

%% 柱状图 MAE MAPE RMSE 柱状图适合量纲差别不大的
color=    [0.66669    0.1206    0.108
    0.1339    0.7882    0.8588
    0.1525    0.6645    0.1290
    0.8549    0.9373    0.8275   
    0.1551    0.2176    0.8627
    0.7843    0.1412    0.1373
    0.2000    0.9213    0.8176
      0.5569    0.8118    0.7882
       1.0000    0.5333    0.5176];
figure('Units', 'pixels', ...
    'Position', [300 300 660 375]);
plot_data_t=Test_all(:,[1,2,4])';
b=bar(plot_data_t,0.8);
hold on

for i = 1 : size(plot_data_t,2)
    x_data(:, i) = b(i).XEndPoints'; 
end

for i =1:size(plot_data_t,2)
b(i).FaceColor = color(i,:);
b(i).EdgeColor=[0.6353    0.6314    0.6431];
b(i).LineWidth=1.2;
end

for i = 1 : size(plot_data_t,1)-1
    xilnk=(x_data(i, end)+ x_data(i+1, 1))/2;
    b1=xline(xilnk,'--','LineWidth',1.2);
    hold on
end 

ax=gca;
legend(b,str1,'Location','best')
ax.XTickLabels ={'MAE', 'MAPE', 'RMSE'};
set(gca,"FontSize",12,"LineWidth",2)
box off
legend box off

%% 二维图
figure
plot_data_t1=Test_all(:,[1,5])';
MarkerType={'s','o','pentagram','^','v'};
for i = 1 : size(plot_data_t1,2)
   scatter(plot_data_t1(1,i),plot_data_t1(2,i),120,MarkerType{i},"filled")
   hold on
end
set(gca,"FontSize",12,"LineWidth",2)
box off
legend box off
legend(str1,'Location','best')
xlabel('MAE')
ylabel('R2')
grid on



colorList=[12 13 167;
          66 124 231;
          136 12 20;
          231 188 198;
          253 207 158;
          239 164 132;
          182 118 108]./255;



%%
figure('Units', 'pixels', ...
    'Position', [150 150 920 600]);
t = tiledlayout('flow','TileSpacing','compact');
for i=1:length(Test_all(:,1))
nexttile
th1 = linspace(2*pi/length(Test_all(:,1))/2,2*pi-2*pi/length(Test_all(:,1))/2,length(Test_all(:,1)));
r1 = Test_all(:,i)';
[u1,v1] = pol2cart(th1,r1);
M=compass(u1,v1);
for j=1:length(Test_all(:,1))
    M(j).LineWidth = 2;
    M(j).Color = colorList(j,:);

end   
title(str2{i})
set(gca,"FontSize",10,"LineWidth",1)
end
 legend(M,str1,"FontSize",10,"LineWidth",1,'Box','off','Location','southoutside')


