function picp = PICP(T_sim, T_train)

%%  矩阵转置
if size(T_sim, 1) ~= size(T_train, 1)
    T_sim = T_sim';
end

%%  区间覆盖率
RangeForm = [T_sim(:, 1), T_sim(:, end)];
Num = 0;

for i = 1 : length(T_train)
    Num = Num +  (T_train(i) >= RangeForm(i, 1) && T_train(i) <= RangeForm(i, 2));
end

picp = Num / length(T_train);     

end