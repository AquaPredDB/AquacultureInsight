function [CWC,mean_CWC] = CWC_FUN(PINAW,PICP,Eta,Beta)
   % CWC
    PINAW = PINAW';
    PICP = PICP';
    for i = 1:numel(PINAW)
        if PICP(i) < Beta(i)
            Gamma(i) = 1;
        else 
            Gamma(i) = 0;
        end
    end
    for m = 1:numel(PINAW)
        CWC(m) = PINAW(m)+Gamma(m)*exp(-Eta*(PICP(m)-Beta(m)));
    end
    mean_CWC = mean(CWC);
end

