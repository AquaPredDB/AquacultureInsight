function [PINAW,mean_PINAW] = PINAW_FUN(Lower,Upper,Real)
    PINAW = sum(Upper-Lower)/(length(Real)*(max(Real)-min(Real)));
    mean_PINAW = mean(PINAW);
end

