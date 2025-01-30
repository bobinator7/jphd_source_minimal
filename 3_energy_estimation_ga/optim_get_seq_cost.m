function cost_ret = optim_get_seq_cost(no_pe, pe_seq, seq_type, layer_struct)
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
cost_ret = zeros(size(pe_seq,1),1);
k_vec = [layer_struct.dat(3:4) layer_struct.ch_in layer_struct.ch_out];
i_vec = [(layer_struct.dim_in + 2*layer_struct.dat(7:8)) layer_struct.ch_in];
o_vec = [layer_struct.dim_out layer_struct.ch_out];

%% loop over cell array with pe_seq configuration
for ii = 1:size(pe_seq,1)

    %% test seq size
    if numel(pe_seq{ii}) ~= test_size
        assert(false,'Function GET_SEQ_COST Error: Sequence array does not match layer configuration!');
    end

    % get initial configuration
    td_conf_conv = {zeros(i_vec) zeros(k_vec) zeros(o_vec)};
    dd_conf_current = {zeros(i_vec) zeros(k_vec) zeros(o_vec)};

    pingpong = 1;
    cost = 0;
    for idx = 1:no_pe:(numel(pe_seq{ii}))

        % get next iteration configuration
        idx_end = idx+no_pe-1;
        if idx_end > numel(pe_seq{ii})
            idx_end = numel(pe_seq{ii});
        end
        
        td_conf_conv{seq_type}(:) = 0;
        td_conf_conv{seq_type}(pe_seq{ii}(idx:idx_end)) = 1;
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
        it_cost = i_cost + k_cost + o_cost;
        cost = cost + it_cost;

        % update current configuration
        %dd_conf_current = dd_conf_next;
        pingpong = ~pingpong;
    end

    cost_ret(ii) = cost;
end

end