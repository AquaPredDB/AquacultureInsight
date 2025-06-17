clc,clear
data = xlsread("数据1.xlsx");
%%
Fcst = data(:,1);
Real = data(:,2);
Step = (1:size(data,1))';
%%
z = [0.95;0.9;0.85;0.8;0.75;0.7;0.5]; %分位数

error_test = Fcst-Real;
%%
for m = 1:7
    [~,stats(:,m)]=INTERVAL_FUN(Step,error_test,z(m),3,150);
end
%%
for m = 1:7
    Lower(:,m) = Fcst - stats(m).yfitci(:,1);
    Upper(:,m) = Fcst + stats(m).yfitci(:,2);
end
%%
PlotProbability(Fcst,Real,numel(Step),Lower,Upper,1,150,...
    '数据',[1 1 1],[ 0    0.5451    0.5451],[1 0 0],[1 0 0]); %概率绘图

% 预测值颜色，

%%
% PINAW
[PINAW,mean_PINAW] = PINAW_FUN(Lower,Upper,Real);
%%
% PICP
[PICP,mean_PICP] = PICP_FUN(Lower,Upper,Real);
%%
% CWC
beta = [0.95;0.9;0.85;0.8;0.75;0.7;0.5]; %置信水平
eta = 50; %惩罚参数
[CWC,mean_CWC] = CWC_FUN(PINAW,PICP,eta,beta);


