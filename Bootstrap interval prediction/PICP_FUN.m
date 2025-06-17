function [PICP,mean_PICP] = PICP_FUN(Lower,Upper,Real)
    temp = zeros(size(Lower,2),1);
    for i = 1:size(Lower,2)
        for j = 1:length(Real)
            if Lower(j,i)<=Real(j)&&Upper(j,i)>=Real(j)
                temp(i,:) = temp(i,:)+1;
                count_picp(:,i) = temp(i,:);
            end
        end  
    end
    PICP = count_picp/length(Real);
    mean_PICP = mean(PICP);
end

