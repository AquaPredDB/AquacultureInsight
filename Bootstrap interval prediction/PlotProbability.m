function PlotProbability(TestOutputs,RealOutputs,Num_Fcst,Lower,Upper,llimit,rlimit,Name,FColor,IColor,RColor,TColor)
    %% 区间估计
    figure; %概率绘图
    transparence = [0;0;0;0]; %图窗透明度
    x = 1:Num_Fcst;
    h = gca;
    hp = patch([x(1),x(end),x(end),x(1)],...
    [min(min(Lower)),min(min(Lower)),max(max(Upper)),max(max(Upper))],FColor,'FaceVertexAlphaData',transparence,'FaceAlpha',"interp");
    uistack(hp,'bottom');
    hold on
    n = [0.2;0.21;0.22;0.23;0.3;0.5;0.6]; %区间透明度
    for j = 1:7
        window(j)=fill([x,fliplr(x)],[Lower(:,j)',fliplr(Upper(:,j)')],IColor,'FaceAlpha',n(j));
        window(j).EdgeColor = 'none';
        hold on
        plot(Upper(:,j),'Marker',"none","LineStyle","none","Tag",'none',"Visible","off");
        hold on
        plot(Lower(:,j),'Marker',"none","LineStyle","none","Tag",'none',"Visible","off");
        hold on
    end
    plot(RealOutputs,'*','MarkerSize',4,'Color',RColor);
    hold on
    plot(TestOutputs,'Color',TColor,'LineWidth',1.5);
    hold on
    xlim([llimit rlimit]);
    ylim([min(min(Lower)) max(max(Upper))]);
    xlabel('采样点',"FontSize",10,"FontWeight","bold");
    ylabel('数据',"FontSize",10,"FontWeight","bold");
    legend('','95%置信区间',"","",'90%置信区间',"","",'85%置信区间',"","",...
    '80%置信区间',"","",'75%置信区间',"","",'70%置信区间',"","",...
    '50%置信区间',"","",'真实值',strcat(Name,'预测值'));
    grid on
end

