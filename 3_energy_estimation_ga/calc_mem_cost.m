function [res,out_arch] = calc_mem_cost(arch, energy_struct, hw_model)
%CALC_MEM_COST
%
% (Usage)
%
% (Examples)
%
% (See also)

% $Author: Johnson Loh $  $Date: 2019/12/18 $ $Revision: 0.1 $
% Copyright: 

BATCH_SIZE = 1;%128;

res = 0;
for ii = 1:size(arch,2)
    switch hw_model.type
        case 'min'
            if strcmp(arch(ii).type,'Data')
                num_offchip_read = BATCH_SIZE * arch(ii).mem_actout + sum([arch.mem_paraminternal]);
                num_offchip_write = 0; 
                num_onchip_read = 0;
                num_onchip_write = 0;                
            else
                num_offchip_read = 0;
                num_offchip_write = 0; 
                num_onchip_read = BATCH_SIZE * (arch(ii).mem_paraminternal + prod(arch(ii).dim_in) * arch(ii).ch_in);
                num_onchip_write = BATCH_SIZE * arch(ii).mem_actout;
            end
        case 'max'
            switch arch(ii).type
                case 'Conv2d'
                    num_offchip_read = BATCH_SIZE * arch(ii).op_mac * 3;
                    num_offchip_write = BATCH_SIZE * arch(ii).op_mac;
                    num_onchip_read = BATCH_SIZE * arch(ii).op_mac * 3;
                    num_onchip_write = BATCH_SIZE * arch(ii).op_mac;
                case 'Linear'
                    num_offchip_read = BATCH_SIZE * arch(ii).op_mac * 3;
                    num_offchip_write = BATCH_SIZE * arch(ii).op_mac;
                    num_onchip_read = BATCH_SIZE * arch(ii).op_mac * 3;
                    num_onchip_write = BATCH_SIZE * arch(ii).op_mac;
                case 'MaxPool2d' % 'AdaptiveAvgPool2d'
                    num_offchip_read = BATCH_SIZE * arch(ii).mem_actout * arch(ii).dat(1).^2;
                    num_offchip_write = BATCH_SIZE * arch(ii).mem_actout;
                    num_onchip_read = BATCH_SIZE * arch(ii).mem_actout * arch(ii).dat(1).^2;
                    num_onchip_write = BATCH_SIZE * arch(ii).mem_actout;
                case 'ReLU'
                    num_offchip_read = BATCH_SIZE * arch(ii).mem_actout;
                    num_offchip_write = BATCH_SIZE * arch(ii).mem_actout;
                    num_onchip_read = BATCH_SIZE * arch(ii).mem_actout;
                    num_onchip_write = BATCH_SIZE * arch(ii).mem_actout;
                otherwise
                    num_offchip_read = 0;
                    num_offchip_write = 0;
                    num_onchip_read = 0;
                    num_onchip_write = 0;
            end
            
        otherwise
            switch arch(ii).type
                case 'Conv2d'
                    [num_onchip_read,num_onchip_write,num_offchip_read,num_offchip_write] = mem_conv_accesses(hw_model.no_pe, hw_model.dataflow_strategy, arch(ii));
                    
                    % on-chip accesses
                    num_onchip_read = BATCH_SIZE * num_onchip_read;
                    num_onchip_write = BATCH_SIZE * num_onchip_write;

                    % off-chip accesses
                    num_offchip_read = BATCH_SIZE * num_offchip_read;
                    num_offchip_write = BATCH_SIZE * num_offchip_write;
                    
                case 'Linear'
                    switch hw_model.dataflow_strategy
                        case 1
                            % on-chip accesses
                            num_weight_access_onchip = arch(ii).mem_paraminternal;
                            num_ifmap_access_onchip = prod([arch(ii).dim_in arch(ii).ch_in]);
                            num_psum_read_onchip = prod([arch(ii).dim_out arch(ii).ch_out]) * ceil(prod([arch(ii).dim_in arch(ii).ch_in]) / hw_model.no_pe);
                            num_psum_write_onchip = prod([arch(ii).dim_out arch(ii).ch_out]) * ceil(prod([arch(ii).dim_in arch(ii).ch_in]) / hw_model.no_pe);
                            
                            % off-chip accesses
                            tile = hw_model.mem_no_vals - prod([arch(ii).dim_out arch(ii).ch_out]);
                            if tile < 0
                                assert(false,'Function CALC_MEM_COST Error: On-Chip Memory too small!');
                            end
                            
                            num_weight_access_offchip = arch(ii).mem_paraminternal;
                            num_ifmap_access_offchip = prod([arch(ii).dim_in arch(ii).ch_in]);
                            num_psum_read_offchip = 0;
                            num_psum_write_offchip = prod([arch(ii).dim_out arch(ii).ch_out]);

                        case 2
                            tile = hw_model.mem_no_vals - prod([arch(ii).dim_out arch(ii).ch_out]) - prod([arch(ii).dim_in arch(ii).ch_in]);
                            if tile < 0
                                assert(false,'Function CALC_MEM_COST Error: On-Chip Memory too small!');
                            end
                            
                            % on-chip accesses
                            num_weight_access_onchip = arch(ii).mem_paraminternal;
                            num_ifmap_access_onchip = arch(ii).mem_paraminternal;
                            num_psum_read_onchip = prod([arch(ii).dim_out arch(ii).ch_out]) * max(ceil(arch(ii).num_val_param_per_out / hw_model.no_pe),1);
                            num_psum_write_onchip = prod([arch(ii).dim_out arch(ii).ch_out]) * max(ceil(arch(ii).num_val_param_per_out / hw_model.no_pe),1);
                            
                            % off-chip accesses
                            num_weight_access_offchip = arch(ii).mem_paraminternal;
                            num_ifmap_access_offchip = prod([arch(ii).dim_in arch(ii).ch_in]);
                            num_psum_read_offchip = 0;
                            num_psum_write_offchip = prod([arch(ii).dim_out arch(ii).ch_out]);
                        case 3
                            tile = hw_model.mem_no_vals - prod([arch(ii).dim_in arch(ii).ch_in]);
                            if tile < 0
                                assert(false,'Function CALC_MEM_COST Error: On-Chip Memory too small!');
                            end
                            
                            % on-chip accesses
                            num_weight_access_onchip = arch(ii).mem_paraminternal;
                            num_ifmap_access_onchip = arch(ii).mem_paraminternal;
                            num_psum_read_onchip = prod([arch(ii).dim_out arch(ii).ch_out]);
                            num_psum_write_onchip = prod([arch(ii).dim_out arch(ii).ch_out]);
                            
                            % off-chip accesses
                            num_weight_access_offchip = arch(ii).mem_paraminternal;
                            num_ifmap_access_offchip = prod([arch(ii).dim_in arch(ii).ch_in]);
                            num_psum_read_offchip = 0;
                            num_psum_write_offchip = prod([arch(ii).dim_out arch(ii).ch_out]);
                        otherwise
                            assert(false,'Function CALC_MEM_COST Error: Sequence type not defined!');
                    end
                    
                    num_onchip_read = BATCH_SIZE * (num_psum_read_onchip + num_weight_access_onchip + num_ifmap_access_onchip);
                    num_onchip_write = BATCH_SIZE * num_psum_write_onchip;
                    
                    num_offchip_read = BATCH_SIZE * (num_psum_read_offchip + num_weight_access_offchip + num_ifmap_access_offchip);
                    num_offchip_write = BATCH_SIZE * num_psum_write_offchip;
%                     num_offchip_read = num_onchip_read;
%                     num_offchip_write = num_onchip_write;

                case {'MaxPool2d', 'ReLU'} % 'AdaptiveAvgPool2d'
                    num_offchip_read = BATCH_SIZE * arch(ii).mem_actout;
                    num_offchip_write = BATCH_SIZE * arch(ii).mem_actout;
                    num_onchip_read = BATCH_SIZE * arch(ii).mem_actout;
                    num_onchip_write = BATCH_SIZE * arch(ii).mem_actout;
                otherwise
                    num_offchip_read = 0;
                    num_offchip_write = 0;
                    num_onchip_read = 0;
                    num_onchip_write = 0;
            end
    end
    arch(ii).e_mem_layer = (num_offchip_read * energy_struct.mem_offchip_read ...
                           + num_offchip_write * energy_struct.mem_offchip_write ...
                           + num_onchip_read * energy_struct.mem_onchip_read ...
                           + num_onchip_write * energy_struct.mem_onchip_write);
                       
end

out_arch = arch;

end