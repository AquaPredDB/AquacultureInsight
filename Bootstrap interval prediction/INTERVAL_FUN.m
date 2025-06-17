function [p,stats]=INTERVAL_FUN(x,y,tau,order,Nboot)

warning off
if nargin<3
    error('输入参数不足！');
end
if nargin<4, order=[]; end
if nargin<5, Nboot=200; end

if (tau<=0)|(tau>=1),
    error('分位数必须介于0和1之间！')
end

if size(x,1)~=size(y,1)
    error('样本点数量必须相同！');
end

if numel(y)~=size(y,1)
    error('因变量序列必须为向量！')
end

if size(x,2)==1
    if isempty(order)
        order=1;
    end
    %建立范德蒙德矩阵
    if order>0
        x(:,order+1)=1;
    else
        order=abs(order);
    end
    x(:,order)=x(:,1); 
    for ii=order-1:-1:1
        x(:,ii)=x(:,order).*x(:,ii+1);
    end
elseif isempty(order)
    order=1; 
else
    error('不能同时使用多列输入来指定回归阶数！');
end


pmean=x\y; %最小二乘
% if all(x(:,end)==1)
%     r=y-x*pmean;
%     pmean(end)=pmean(end)+prctile(r,tau*100);
% end

rho=@(r)sum(abs(r.*(tau-(r<0))));

p=fminsearch(@(p)rho(y-x*p),pmean);




if nargout>1
    %采用 Ｂｏｏｔｓｔｒａｐ方法分析模型的预测误差分布
    
    yfit=x*p;
    resid=y-yfit;
    
    options = optimset('MaxFunEvals',1000,'MaxIter',1000);
    stats.pboot=bootstrp(Nboot,@(bootr)fminsearch(@(p)rho(yfit+bootr-x*p),p,options)', resid);
    stats.pse=std(stats.pboot);
    
    qq=zeros(size(x,1),Nboot);
    for ii=1:Nboot
        qq(:,ii)=x*stats.pboot(ii,:)';
    end
    stats.yfitci=prctile(qq',[2.5 97.5])';
    
end








