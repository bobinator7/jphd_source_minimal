function arch = calc_key_param(net_arch)
%CALC_KEY_PARAM
%
% (Usage)
%
% (Examples)
%
% (See also)

% $Author: Johnson Loh $  $Date: 2019/12/18 $ $Revision: 0.1 $
% Copyright: 

for ii = 1:size(net_arch,2)
    switch net_arch(ii).type
%         case 'AdaptiveAvgPool2d'
%             net_arch(ii).dim_in = net_arch(ii-1).dim_out;
%             net_arch(ii).ch_in = net_arch(ii-1).ch_out;
%             net_arch(ii).dim_out = net_arch(ii).dat(1:2);
%             net_arch(ii).ch_out = net_arch(ii-1).ch_out;
% 
%             net_arch(ii).op_mac = 0;
%             net_arch(ii).op_compare = 0;
%             net_arch(ii).op_add = 0;
%             net_arch(ii).op_div = 0;
% 
%             net_arch(ii).mem_actout = prod(net_arch(ii).dim_out) .* net_arch(ii).ch_out;
%             net_arch(ii).mem_paraminternal = 0;
        case 'Conv2d'
            net_arch(ii).dim_in = net_arch(ii-1).dim_out;
            net_arch(ii).ch_in = net_arch(ii).dat(1);
            net_arch(ii).dim_out = floor((net_arch(ii-1).dim_out + net_arch(ii).dat(7:8) .* 2 - net_arch(ii).dat(3:4)) ./ net_arch(ii).dat(5:6)) + ones(1,2);
            net_arch(ii).ch_out = net_arch(ii).dat(2);
            
            net_arch(ii).num_val_in = prod([(net_arch(ii).dim_in + 2*net_arch(ii).dat(7:8)) net_arch(ii).ch_in]);%prod(net_arch(ii).dim_in) * net_arch(ii).ch_in;
            net_arch(ii).num_val_out = prod(net_arch(ii).dim_out) * net_arch(ii).ch_out;
            net_arch(ii).num_val_param_per_out = prod(net_arch(ii).dat(3:4)) .* net_arch(ii).ch_in;
            
            net_arch(ii).op_mac = prod(net_arch(ii).dim_out) .* net_arch(ii).ch_out .* prod(net_arch(ii).dat(3:4)) .* prod(net_arch(ii).dat(1));
            net_arch(ii).op_compare = 0;
            net_arch(ii).op_add = 0;
            net_arch(ii).op_div = 0;

            net_arch(ii).mem_actout = prod(net_arch(ii).dim_out) .* net_arch(ii).ch_out;
            net_arch(ii).mem_paraminternal = prod(net_arch(ii).dat(3:4)) .* net_arch(ii).ch_in * net_arch(ii).ch_out;
        case 'Data'
            net_arch(ii).dim_in = net_arch(ii).dat(1:2);
            net_arch(ii).ch_in = net_arch(ii).dat(3);
            net_arch(ii).dim_out = net_arch(ii).dat(1:2);
            net_arch(ii).ch_out = net_arch(ii).dat(3);
            
            net_arch(ii).num_val_in = prod(net_arch(ii).dim_in) * net_arch(ii).ch_in;
            net_arch(ii).num_val_out = prod(net_arch(ii).dim_out) * net_arch(ii).ch_out;
            net_arch(ii).num_val_param_per_out = 0;

            net_arch(ii).op_mac = 0;
            net_arch(ii).op_compare = 0;
            net_arch(ii).op_add = 0;
            net_arch(ii).op_div = 0;

            net_arch(ii).mem_actout = prod(net_arch(ii).dim_out) .* net_arch(ii).ch_out;
            net_arch(ii).mem_paraminternal = 0;
        case 'Dropout'
            net_arch(ii).dim_in = net_arch(ii-1).dim_out;
            net_arch(ii).ch_in = net_arch(ii-1).ch_out;
            net_arch(ii).dim_out = net_arch(ii-1).dim_out;
            net_arch(ii).ch_out = net_arch(ii-1).ch_out;
            
            net_arch(ii).num_val_in = prod(net_arch(ii).dim_in) * net_arch(ii).ch_in;
            net_arch(ii).num_val_out = prod(net_arch(ii).dim_out) * net_arch(ii).ch_out;
            net_arch(ii).num_val_param_per_out = 0;

            net_arch(ii).op_mac = 0;
            net_arch(ii).op_compare = 0;
            net_arch(ii).op_add = 0;
            net_arch(ii).op_div = 0;

            net_arch(ii).mem_actout = prod(net_arch(ii).dim_out) .* net_arch(ii).ch_out;
            net_arch(ii).mem_paraminternal = 0;
        case 'Linear'
            net_arch(ii).dim_in = net_arch(ii-1).dim_out;
            net_arch(ii).ch_in = net_arch(ii-1).ch_out;
            net_arch(ii).dim_out = ones(1,2);
            net_arch(ii).ch_out = net_arch(ii).dat(2);
            
            net_arch(ii).num_val_in = prod(net_arch(ii).dim_in) * net_arch(ii).ch_in;
            net_arch(ii).num_val_out = prod(net_arch(ii).dim_out) * net_arch(ii).ch_out;
            net_arch(ii).num_val_param_per_out = prod(net_arch(ii).dim_in) .* net_arch(ii).ch_in;

            net_arch(ii).op_mac = prod(net_arch(ii).dim_out) .* net_arch(ii).ch_out .* prod(net_arch(ii).dim_in) .* net_arch(ii).ch_in;
            net_arch(ii).op_compare = 0;
            net_arch(ii).op_add = 0;
            net_arch(ii).op_div = 0;

            net_arch(ii).mem_actout = prod(net_arch(ii).dim_out) .* net_arch(ii).ch_out;
            net_arch(ii).mem_paraminternal = prod(net_arch(ii).dim_in) .* net_arch(ii).ch_in .* prod(net_arch(ii).dim_out) .* net_arch(ii).ch_out;
        case 'MaxPool2d'
            net_arch(ii).dim_in = net_arch(ii-1).dim_out;
            net_arch(ii).ch_in = net_arch(ii-1).ch_out;
            net_arch(ii).dim_out = floor((net_arch(ii-1).dim_out + [net_arch(ii).dat(3) net_arch(ii).dat(3)] * 2 - [net_arch(ii).dat(1) net_arch(ii).dat(1)] )./ (net_arch(ii).dat(2) * ones(1,2)))  + ones(1,2);
            net_arch(ii).ch_out = net_arch(ii-1).ch_out;
            
            net_arch(ii).num_val_in = prod(net_arch(ii).dim_in) * net_arch(ii).ch_in;
            net_arch(ii).num_val_out = prod(net_arch(ii).dim_out) * net_arch(ii).ch_out;
            net_arch(ii).num_val_param_per_out = 0;

            net_arch(ii).op_mac = 0;
            net_arch(ii).op_compare = prod(net_arch(ii).dim_out) .* net_arch(ii).ch_out .* net_arch(ii).dat(1).^2;
            net_arch(ii).op_add = 0;
            net_arch(ii).op_div = 0;

            net_arch(ii).mem_actout = prod(net_arch(ii).dim_out) .* net_arch(ii).ch_out;
            net_arch(ii).mem_paraminternal = 0;
        case 'ReLU'
            net_arch(ii).dim_in = net_arch(ii-1).dim_out;
            net_arch(ii).ch_in = net_arch(ii-1).ch_out;
            net_arch(ii).dim_out = net_arch(ii-1).dim_out;
            net_arch(ii).ch_out = net_arch(ii-1).ch_out;
            
            net_arch(ii).num_val_in = prod(net_arch(ii).dim_in) * net_arch(ii).ch_in;
            net_arch(ii).num_val_out = prod(net_arch(ii).dim_out) * net_arch(ii).ch_out;
            net_arch(ii).num_val_param_per_out = 0;

            net_arch(ii).op_mac = 0;
            net_arch(ii).op_compare = prod(net_arch(ii).dim_out) .* net_arch(ii).ch_out;
            net_arch(ii).op_add = 0;
            net_arch(ii).op_div = 0;

            net_arch(ii).mem_actout = prod(net_arch(ii).dim_out) .* net_arch(ii).ch_out;
            net_arch(ii).mem_paraminternal = 0;
        otherwise
            
    end
end

arch = net_arch;

end