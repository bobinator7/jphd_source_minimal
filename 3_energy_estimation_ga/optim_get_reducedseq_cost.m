function [read_onchip,write_onchip,read_offchip,write_offchip] = optim_get_reducedseq_cost(no_pe, seq_type, layer_struct)
%OPTIM_GET_REDUCEDSEQ_COST
%
% (Usage)
%
% (Examples)
%
% (See also)

% $Author: Johnson Loh $  $Date: 2020/01/21 $ $Revision: 0.1 $
% Copyright: 

%% check td array size match layer struct
% switch seq_type
%     case 1
%         test_size = prod(layer_struct.dim_in + 2*layer_struct.dat(7:8));
%     case 2
%         test_size = prod(layer_struct.dat(3:4));
%     case 3
%         test_size = prod(layer_struct.dim_out);
%     otherwise
%         assert(false,'Function GET_SEQ_COST Error: Sequence type not defined!');
% end

%% 
k_vec = [layer_struct.dat(3:4) layer_struct.ch_in layer_struct.ch_out];
i_vec = [(layer_struct.dim_in + 2*layer_struct.dat(7:8)) layer_struct.ch_in];
o_vec = [layer_struct.dim_out layer_struct.ch_out];

%% test seq size
switch seq_type
    case 1
        test_size = prod(layer_struct.dim_in + 2*layer_struct.dat(7:8));
        pe_seq = 1:test_size;
    case 2
        test_size = prod(layer_struct.dat(3:4));
        pe_seq = 1:test_size;
    case 3
        test_size = prod(layer_struct.dim_out);
        pe_seq = 1:test_size;
    otherwise
        assert(false,'Function OPTIM_GET_REDUCEDSEQ_COST Error: Sequence type not defined!');
end

if no_pe > test_size
    assert(false,'Function OPTIM_GET_REDUCEDSEQ_COST Error: Size of PE array is larger than 2D window! Next tiling hierarchy (Tiling across channels) is possible!');
end

% get initial configuration
td_conf_conv = {zeros(i_vec) zeros(k_vec) zeros(o_vec)};
dd_conf_current = {zeros(i_vec) zeros(k_vec) zeros(o_vec)};

pingpong = 1;
read_offchip = 0;
write_offchip = 0;
read_onchip = 0;
write_onchip = 0;
for idx = 1:no_pe:(numel(pe_seq))

    % get next iteration configuration
    idx_end = idx+no_pe-1;
    if idx_end > numel(pe_seq)
        idx_end = numel(pe_seq);
    end

    td_conf_conv{seq_type}(:) = 0;
    td_conf_conv{seq_type}(pe_seq(idx:idx_end)) = 1;
    if pingpong
        dd_conf_next = get_dd_conf(td_conf_conv, layer_struct);

        % get next iteration new values
        i_cost_offchip = nnz((dd_conf_next{1} - dd_conf_current{1}) > 0);
        k_cost_offchip = nnz((dd_conf_next{2} - dd_conf_current{2}) > 0);
        o_cost_offchip = nnz((dd_conf_next{3} - dd_conf_current{3}) > 0);
        
        i_cost_onchip = nnz(dd_conf_next{1});
        k_cost_onchip = nnz(dd_conf_next{2});
        o_cost_onchip = nnz(dd_conf_next{3});
    else
        dd_conf_current = get_dd_conf(td_conf_conv, layer_struct);

        % get next iteration new values
        i_cost_offchip = nnz((dd_conf_current{1} - dd_conf_next{1}) > 0);
        k_cost_offchip = nnz((dd_conf_current{2} - dd_conf_next{2}) > 0);
        o_cost_offchip = nnz((dd_conf_current{3} - dd_conf_next{3}) > 0);
        
        i_cost_onchip = nnz(dd_conf_current{1});
        k_cost_onchip = nnz(dd_conf_current{2});
        o_cost_onchip = nnz(dd_conf_current{3});
    end

    % calculate cost onchip
    read_onchip = read_onchip + i_cost_onchip + k_cost_onchip + o_cost_onchip;
    write_onchip = write_onchip + o_cost_onchip;

    % calculate cost offchip (atm: 1 per overlap)
    read_offchip = read_offchip + i_cost_offchip + k_cost_offchip + o_cost_offchip;
    write_offchip = write_offchip + o_cost_offchip;

    % update current configuration
    %dd_conf_current = dd_conf_next;
    pingpong = ~pingpong;
end


end