function pop = optim_create_permutations(NVARS,FitnessFcn,options)
%OPTIMS_CREATE_PERMUTATIONS
%
% (Usage)
%
% (Examples)
%
% (See also) https://www.mathworks.com/help/gads/custom-data-type-optimization-using-ga.html

% $Author: Johnson Loh $  $Date: 2020/01/28 $ $Revision: 0.1 $
% Copyright: 

totalPopulationSize = sum(options.PopulationSize);
n = NVARS;
pop = cell(totalPopulationSize,1);
for i = 1:totalPopulationSize
    
%     if i == 1
%         pop{i} = 1:n;
%     elseif i == 2
%         pop{i} = n:-1:1;
%     end
    
    pop{i} = randperm(n); 
end

end