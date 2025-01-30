function xoverKids  = optim_crossover_permutation(parents,options,NVARS, ...
    FitnessFcn,thisScore,thisPopulation)
%OPTIM_CROSSOVER_PERMUTATION
%
% (Usage)
%
% (Examples)
%
% (See also) https://www.mathworks.com/help/gads/custom-data-type-optimization-using-ga.html

% $Author: Johnson Loh $  $Date: 2020/01/28 $ $Revision: 0.1 $
% Copyright: 

nKids = length(parents)/2;
xoverKids = cell(nKids,1); % Normally zeros(nKids,NVARS);
index = 1;

for i=1:nKids
    % here is where the special knowledge that the population is a cell
    % array is used. Normally, this would be thisPopulation(parents(index),:);
    parent = thisPopulation{parents(index)};
    index = index + 2;

    % Flip a section of parent1.
    p1 = ceil((length(parent) -1) * rand);
    p2 = p1 + ceil((length(parent) - p1- 1) * rand);
    child = parent;
    child(p1:p2) = fliplr(child(p1:p2));
    xoverKids{i} = child; % Normally, xoverKids(i,:);
end