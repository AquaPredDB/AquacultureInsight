%% 闲鱼：深度学习与智能算法
%% 唯一官方店铺：https://mbd.pub/o/author-aWWbm3BtZw==
%% 微信公众号：强盛机器学习，关注公众号获得更多免费代码！
classdef FlipLayer < nnet.layer.Layer
%%  数据翻转
    methods
        function layer = FlipLayer(name)
            layer.Name = name;
        end
        function Y = predict(~, X)
                 Y = flip(X, 3);
        end
    end
end
%