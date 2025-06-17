%%  清空环境变量
warning off             % 关闭报警信息
close all               % 关闭开启的图窗
clear                   % 清空变量
clc  

%% 导入数据
data =  readmatrix('风电场预测.xlsx');
data = data(5665:8640,13);  %选取3月份数据,第13列为功率数据，单变量的意思是只选取这一列的变量
nn =12;   %预测未来12个时刻的数据
[h1,l1]=data_process(data,24,nn);   %步长为24，采用前24个时刻的温度预测第25-36个时刻的温度
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



hiddennum_best = 30;
net0=newff(p_train, t_train,hiddennum_best,{'tansig','purelin'},'trainlm');% 建立模型
%网络参数配置
net0.trainParam.epochs=1000;         % 训练次数，这里设置为1000次
net0.trainParam.lr=0.01;                   % 学习速率，这里设置为0.01
net0.trainParam.goal=0.00001;                    % 训练目标最小误差，这里设置为0.0001
net0.trainParam.show=25;                % 显示频率，这里设置为每训练25次显示一次
net0.trainParam.mc=0.01;                 % 动量因子
net0.trainParam.min_grad=1e-6;       % 最小性能梯度
net0.trainParam.max_fail=6;               % 最高失败次数
% net0.trainParam.showWindow = false;
% net0.trainParam.showCommandLine = false;            %隐藏仿真界面
%开始训练
net0=train(net0,p_train, t_train);


t_sim= sim(net0, p_test); 

%  数据反归一化

T_sim = mapminmax('reverse', t_sim, ps_output);

%% 比较算法预测值
str={'真实值','BP'};
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



str={'真实值','BP'};
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
