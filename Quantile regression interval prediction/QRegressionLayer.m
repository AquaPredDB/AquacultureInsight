classdef QRegressionLayer < nnet.layer.RegressionLayer
    
    % �Զ���ع��
    properties
        tau
    end

    methods

        function layer = QRegressionLayer(name, tau)
            
            % �����ع�㣬�������ƺͲ���
            layer.tau = tau;                       % ������������
            layer.Name = name;                     % �������������
            layer.Description = 'quantile error';  % ��������������

        end
        
        function loss = forwardLoss(layer, Y, T)
            % �õ���ʵֵT��Ԥ��ֵY
            % ����ƽ���������
            R = size(Y, 1);
            quantileError = sum(max(layer.tau * (T - Y), (1 - layer.tau) * (Y - T))) / R;  % �в��ֵ㲻�ɵ������ԸĽ�
            N = size(Y, 3);
            loss = sum(quantileError) / N;
        end

        % ��ʧ�����ݶ��½������
        function dLdY = backwardLoss(layer, Y, T)
           
            dLdY =  single(-layer.tau * (T - Y >= 0) + (1 - layer.tau) * (Y - T >= 0));
                    
        end

    end

end