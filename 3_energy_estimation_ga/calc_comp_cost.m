function [res,out_arch] = calc_comp_cost(arch, energy_struct)
%CALC_COMP_COST
%
% (Usage)
%
% (Examples)
%
% (See also)

% $Author: Johnson Loh $  $Date: 2019/12/18 $ $Revision: 0.1 $
% Copyright: 

BATCH_SIZE = 128;

res = 0;
for ii = 1:size(arch,2)
    e_mac = arch(ii).op_mac .* energy_struct.op_mac;
    e_compare = arch(ii).op_compare .* energy_struct.op_compare;
    e_add = arch(ii).op_add .* energy_struct.op_add;
    e_div = arch(ii).op_div .* energy_struct.op_div;
    arch(ii).e_comp_layer = e_mac + e_compare + e_add + e_div;
    res = res + arch(ii).e_comp_layer;
end
res = BATCH_SIZE * res;
out_arch = arch;

end