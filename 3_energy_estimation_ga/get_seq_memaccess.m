function [cost_total, cost_input, cost_weight, cost_output] = get_seq_memaccess(no_pe, pe_seq, seq_type, layer_struct)
%OPTIM_GET_SEQ_COST
%
% (Usage)
%
% (Examples)
%
% (See also)

% $Author: Johnson Loh $  $Date: 2020/01/21 $ $Revision: 0.1 $
% Copyright: 

%% check td array size match layer struct
switch seq_type
    case 1
        test_size = layer_struct.num_val_in;
    case 2
        test_size = layer_struct.mem_paraminternal;
    case 3
        test_size = layer_struct.num_val_out;
    otherwise
        assert(false,'Function GET_SEQ_COST Error: Sequence type not defined!');
end

%%
switch layer_struct.type
    case 'Conv2D'
        k_vec = [layer_struct.dat(3:4) layer_struct.ch_in layer_struct.ch_out];
        i_vec = [(layer_struct.dim_in + 2*layer_struct.dat(7:8)) layer_struct.ch_in];
        o_vec = [layer_struct.dim_out layer_struct.ch_out];
    case 'Linear'
        k_vec = [layer_struct.num_val_in layer_struct.num_val_out];
        i_vec = [layer_struct.num_val_in 1];
        o_vec = [1 layer_struct.num_val_out];
    otherwise
        assert(false,'Function GET_SEQ_COST Error: Layer type not supported!');
end


%% test seq size
switch layer_struct.type
    case 'Conv2D'
        if numel(pe_seq) ~= test_size
            assert(false,'Function GET_SEQ_COST Error: Sequence array does not match layer configuration!');
        end
    case 'Linear'

    otherwise
        assert(false,'Function GET_SEQ_COST Error: Layer type not supported!');
end


% get initial configuration
td_conf_conv = {zeros(i_vec) zeros(k_vec) zeros(o_vec)};
dd_conf_current = {zeros(i_vec) zeros(k_vec) zeros(o_vec)};

pingpong = 1;
cost_input = 0;
cost_weight = 0;
cost_output = 0;    
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
        i_cost = nnz((dd_conf_next{1} - dd_conf_current{1}) > 0);
        k_cost = nnz((dd_conf_next{2} - dd_conf_current{2}) > 0);
        o_cost = nnz((dd_conf_next{3} - dd_conf_current{3}) > 0);
    else
        dd_conf_current = get_dd_conf(td_conf_conv, layer_struct);

        % get next iteration new values
        i_cost = nnz((dd_conf_current{1} - dd_conf_next{1}) > 0);
        k_cost = nnz((dd_conf_current{2} - dd_conf_next{2}) > 0);
        o_cost = nnz((dd_conf_current{3} - dd_conf_next{3}) > 0);
    end

    % calculate cost (atm: 1 per overlap)
    cost_input = cost_input + i_cost;
    cost_weight = cost_weight + k_cost;
    cost_output = cost_output + o_cost;    

    % update current configuration
    %dd_conf_current = dd_conf_next;
    pingpong = ~pingpong;
end

cost_total = cost_input + cost_weight + cost_output;

end