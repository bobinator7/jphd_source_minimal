function mutationChildren = optim_mutate_permutation(parents ,options,NVARS, ...
    FitnessFcn, state, thisScore,thisPopulation,mutationRate)
%OPTIM_MUTATE_PERMUTATION
%
% (Usage)
%
% (Examples)
%
% (See also) https://www.mathworks.com/help/gads/custom-data-type-optimization-using-ga.html

% $Author: Johnson Loh $  $Date: 2020/01/28 $ $Revision: 0.1 $
% Copyright: 

% Here we swap two elements of the permutation
mutationChildren = cell(length(parents),1);% Normally zeros(length(parents),NVARS);
for i=1:length(parents)
    parent = thisPopulation{parents(i)}; % Normally thisPopulation(parents(i),:)
    p = ceil(length(parent) * rand(1,2));
    child = parent;
    child(p(1)) = parent(p(2));
    child(p(2)) = parent(p(1));
    mutationChildren{i} = child; % Normally mutationChildren(i,:)
end