function pimwp = PINAW(T_sim, T_train)

%%  矩阵转置
if size(T_sim, 1) ~= size(T_train, 1)
    T_sim = T_sim';
end

%%  区间平均宽度百分比
pimwp = 1 / length(T_train) * sum((T_sim(:, end) - T_sim(:, 1))...
                                ./ (max(T_train)-min(T_train)));

end