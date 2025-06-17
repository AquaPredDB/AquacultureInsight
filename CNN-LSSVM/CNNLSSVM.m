%%  清空环境变量
warning off             % 关闭报警信息
close all               % 关闭开启的图窗
clear                   % 清空变量
clc  
addpath(genpath(pwd))
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
trainD =  double(reshape(p_train,size(p_train,1),1,1,size(p_train,2)));
testD  =  double(reshape(p_test,size(p_test,1),1,1,size(p_test,2)));
targetD =  t_train;
targetD_test  =  t_test;

%  创建CNN网络，

layers = [
    imageInputLayer([size(p_train,1) 1 1], "Name","sequence")

    convolution2dLayer([3,1],16,'Padding','same')         % 卷积核大小为3*1 生成16个卷积
    batchNormalizationLayer                               % 批归一化层
    reluLayer                                             %relu激活函数

    maxPooling2dLayer([2 1],'Stride',1, "Name", "pool1")  % 最大池化层 大小为3*1 步长为1
    convolution2dLayer([2 1], 32, "Name", "conv_2")       % 卷积核大小为2*1 生成32个卷积
    batchNormalizationLayer                               % 批归一化层
    reluLayer                                             % relu激活层

    maxPooling2dLayer([2 1],'Stride',1, "Name", "pool2")  % 最大池化层 大小为2*2 步长为2


    fullyConnectedLayer(25) % 全连接层神经元
    reluLayer                       %relu激活函数
    fullyConnectedLayer(outdim)      % 输出层神经元
    regressionLayer];%添加回归层，用于计算损失值


%  参数设置
options = trainingOptions('adam', ...                 % 优化算法Adam
    'MaxEpochs', 150, ...                            % 最大训练次数
    'GradientThreshold', 1, ...                       % 梯度阈值
    'InitialLearnRate', 0.01, ...         % 初始学习率
    'LearnRateSchedule', 'piecewise', ...             % 学习率调整
    'LearnRateDropPeriod', 70, ...                   % 训练850次后开始调整学习率
    'LearnRateDropFactor',0.1, ...                    % 学习率调整因子
    'L2Regularization', 0.001, ...         % 正则化参数
    'ExecutionEnvironment', 'cpu',...                 % 训练环境
    'Verbose', 1, ...                                 % 关闭优化过程
    'Plots', 'none');                    % 画出曲线

%  训练
tic
net = trainNetwork(trainD,targetD',layers,options);
toc

%% 提取CNN特征
layer = 'pool2';
p_train = activations(net,trainD,layer,'OutputAs','rows');
p_test  = activations(net,testD, layer,'OutputAs','rows');
%%  类型转换
p_train =  double(p_train); p_test  =  double(p_test);
t_train =  double(t_train); t_test  =  double(t_test);


%随机设定gam和sig2的值
gam = 30;
sig2 = 8500;
model = initlssvm(p_train,t_train','f',[],[],'RBF_kernel');
%实现多分类，要对下列参数进行设置！
model.code = 'changed';
model.codetype = 'code_OneVsOne';
model.gam = gam*ones(1,45);  %这里的45是原始LSSVM在寻优gam和kernel_pars时，自己设定的参数，这个可以不必在意！
model.kernel_pars = sig2*ones(1,45);
model = trainlssvm(model);

t_sim = simlssvm(model,p_test);


%  数据反归一化

T_sim = mapminmax('reverse', t_sim', ps_output);



%% 比较算法预测值
str={'真实值','CNN-LSSVM'};
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



str={'真实值','CNN-LSSVM'};
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

%% 雷达图
figure('Units', 'pixels', ...
    'Position', [150 150 520 500]);
Test_all1=Test_all./sum(Test_all);  %把各个指标归一化到一个量纲
Test_all1(:,end)=1-Test_all(:,end);
RC=radarChart(Test_all1);
str3={'A-MAE','A-MAPE','A-MSE','A-RMSE','1-R2'};
RC.PropName=str3;
RC.ClassName=str1;
RC=RC.draw(); 
RC.legend();
colorList=[12 13 167;
          66 124 231;
          136 12 20;
          231 188 198;
          253 207 158;
          239 164 132;
          182 118 108]./255;
for n=1:RC.ClassNum
    RC.setPatchN(n,'Color',colorList(n,:),'MarkerFaceColor',colorList(n,:))
end


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


