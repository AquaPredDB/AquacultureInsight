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


%% Adaboost增强学习部分
% 样本权重
[mm,nn]=size(P_train);
D(1,:)=ones(1,nn)/nn;

K=10;
for i=1:K

    % ELM创建
    [IW,B,LW,TF,TYPE] = elmtrain(p_train,t_train,80,'sig',0);
  
    %弱预测器训练
    t_sim1 = elmpredict(p_train,IW,B,LW,TF,TYPE);

    % 反归一化
    T_sim1 = mapminmax('reverse',t_sim1,ps_output);
    
    %预测误差
    erroryc(i,:)=T_train-T_sim1;

    %测试数据预测
    t_sim2 = elmpredict(p_test,IW,B,LW,TF,TYPE);
    test_simu(i,:)=mapminmax('reverse',t_sim2,ps_output);% 各个弱预测器预测结果
    
    %调整D值
    Error(i)=0;
    for j=1:nn
        if abs(erroryc(i,j))>0.2  %较大误差
            Error(i)=Error(i)+D(i,j);
            D(i+1,j)=D(i,j)*1.1;
        else
            D(i+1,j)=D(i,j);
        end
    end
    
    %计算弱预测器权重
    at(i)=0.5/exp(abs(Error(i)));
    
    %D值归一化
    D(i+1,:)=D(i+1,:)/sum(D(i+1,:));
end

%% 强预测器预测
at=at/sum(at);

%% 强学习器学习预测结果 %% 来自：公众号《淘个代码》
T_sim2=at*test_simu;                %对测试集进行预测

%% 保存结果 %% 来自：公众号《淘个代码》
ELM_Adaboost_TSIM1 = T_sim1;
ELM_Adaboost_TSIM2 = T_sim2;
save ELM_Adaboost ELM_Adaboost_TSIM1 ELM_Adaboost_TSIM2
save true T_test
%%  计算各项误差参数  %% 来自：公众号《淘个代码》
% 指标计算
disp('…………ELM-Adaboost训练集误差指标…………')
[test_MAE1,test_MAPE1,test_MSE1,test_RMSE1,test_R2_1,test_RPD1] = calc_error(T_train,T_sim1);
fprintf('\n')
disp('…………ELM-Adaboost测试集误差指标…………')
[test_MAE2,test_MAPE2,test_MSE2,test_RMSE2,test_R2_2,test_RPD2]  = calc_error(T_test,T_sim2);
fprintf('\n')

%%  训练集绘图 %% 来自：公众号《淘个代码》
figure('Position',[200,300,1100,300])
plot(1:M,T_train,'b-*','LineWidth',0.1,'MarkerSize',2)
hold on
plot(1:M,T_sim1,'r-o','LineWidth',0.1,'MarkerSize',3)

legend('真实值','ELM-Adaboost预测值')
xlabel('预测样本')
ylabel('预测结果')
string={'训练集预测结果对比';['(R^2 =' num2str(test_R2_1) ' RMSE= ' num2str(test_RMSE1) ' MSE= ' num2str(test_MSE1)  ')'];[ '(MAE= ' num2str(test_MAE1) ' MAPE= ' num2str(test_MAPE1) ' RPD= ' num2str(test_RPD1) ')' ]};
title(string)
% 训练集回归拟合图和误差直方图 来自：公众号《淘个代码》
figure;
plotregression(T_train,T_sim1,['训练集回归图']);
figure;
ploterrhist(T_train-T_sim1,['训练集误差直方图']);


%%  测试集绘图 %% 来自：公众号《淘个代码》
figure('Position',[200,300,1100,300])
plot(1:N,T_test,'b-*','LineWidth',0.1,'MarkerSize',2)
hold on
plot(1:N,T_sim2,'r-o','LineWidth',0.1,'MarkerSize',3)
legend('真实值','ELM-Adaboost预测值')
xlabel('预测样本')
ylabel('预测结果')
string={'测试集预测结果对比';['(R^2 =' num2str(test_R2_2) ' RMSE= ' num2str(test_RMSE2) ' MSE= ' num2str(test_MSE2)  ')'];[ '(MAE= ' num2str(test_MAE2) ' MAPE= ' num2str(test_MAPE2) ' RPD= ' num2str(test_RPD2) ')' ]};
title(string)

figure; %% 来自：公众号《淘个代码》
plotregression(T_test,T_sim2,['测试集回归图']);
figure;
ploterrhist(T_test-T_sim2,['测试集误差直方图']);

%测试集误差图  %% 来自：公众号《淘个代码》
figure('Position',[200,300,1100,300])  %% 来自：公众号《淘个代码》
plot(T_test-T_sim2,'b-*','LineWidth',0.1,'MarkerSize',2)
xlabel('测试集样本编号')
ylabel('预测误差')
title('测试集预测误差')
grid on;
legend('ELM-Adaboost预测输出误差')

%% 来自：公众号《淘个代码》
%微信公众号搜索：淘个代码，获取更多免费代码
%禁止倒卖转售，违者必究！！！！！
%唯一官方店铺：https://mbd.pub/o/author-amqYmHBs/work，其他途径都是骗子！