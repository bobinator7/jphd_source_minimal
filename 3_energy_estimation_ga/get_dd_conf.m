function res_vec = get_dd_conf(td_conf, layer_struct)
%GET_DD_CONF
%
% (Usage)
%
% (Examples)
%
% (See also)

% $Author: Johnson Loh $  $Date: 2020/01/13 $ $Revision: 0.1 $
% Copyright:

%% check td array size match layer struct
check1 = numel(td_conf{1}) ~= layer_struct.num_val_in;
check2 = numel(td_conf{2}) ~= layer_struct.mem_paraminternal;
check3 = numel(td_conf{3}) ~= layer_struct.num_val_out;

if check1 || check2 || check3
    assert(false,'Function GET_DD_CONF Error: Target data size does not match layer configuration');
end

%%
if strcmp(layer_struct.type,'Linear')
    in_dd_conf_from_weight = any(td_conf{2},2);
    in_dd_conf_from_out = ones(layer_struct.num_val_in,1) * any(td_conf{3});
    weight_dd_conf_from_in = repmat(td_conf{1},1,numel(td_conf{3}));
    weight_dd_conf_from_out = repmat(td_conf{3},numel(td_conf{1}),1);
    out_dd_conf_from_in = ones(1,layer_struct.num_val_out) * any(td_conf{1});
    out_dd_conf_from_weight = any(td_conf{2},1);
        
    in_dd_conf = in_dd_conf_from_weight + in_dd_conf_from_out;
    weight_dd_conf = weight_dd_conf_from_in + weight_dd_conf_from_out;
    out_dd_conf = out_dd_conf_from_in + out_dd_conf_from_weight;
elseif strcmp(layer_struct.type,'Conv2d')
    %% function in
    in_td_conf = td_conf{1};
    weight_td_conf = td_conf{2};
    out_td_conf = td_conf{3};

    
    %% layer in
    k_vec = [layer_struct.dat(3:4) layer_struct.ch_in layer_struct.ch_out];
    i_vec = [(layer_struct.dim_in + 2*layer_struct.dat(7:8)) layer_struct.ch_in];
    o_vec = [layer_struct.dim_out layer_struct.ch_out];
    i_stride = layer_struct.dat(5:6);
    
    %% conf init
    in_dd_conf = zeros(i_vec);
    weight_dd_conf = zeros(k_vec);
    out_dd_conf = zeros(o_vec);
    
    %% conf from input
    [row,col] = find(in_td_conf);
    %weight_dd_conf_from_in_usage = zeros(k_vec);
    %out_dd_conf_from_in_usage = zeros(o_vec);
    %weight_dd_conf_from_in = zeros(k_vec);
    %out_dd_conf_from_in = zeros(o_vec);
    
    for ii = 1:size(row,1)
        
        ch_in = mod(floor((col(ii)-1)/i_vec(2)),i_vec(3))+1;
        col_tmp = mod((col(ii)-1),i_vec(2))+1;
        
        % weight dd
        for jj = 1:k_vec(1)
            for kk = 1:k_vec(2)
                row_check = jj:i_stride(2):(i_vec(2)-k_vec(2)+jj);
                col_check = kk:i_stride(1):(i_vec(1)-k_vec(1)+kk);
                check1 = any(row_check(:) == row(ii));
                check2 = any(col_check(:) == col_tmp);
                if check1 && check2
                    %weight_dd_conf_from_in_usage(jj,kk,ch_in,:) = weight_dd_conf_from_in_usage(jj,kk,ch_in,:) + 1;
                    %weight_dd_conf_from_in(jj,kk,ch_in,:) = 1;
                    weight_dd_conf(jj,kk,ch_in,:) = 1;
                end
            end
        end
        
        % out dd
        for jj = 1:o_vec(1)
            for kk = 1:o_vec(2)
                row_check = (jj+(i_stride(2)-1)*(jj-1)):(jj+(i_stride(2)-1)*(jj-1)+k_vec(2)-1);
                col_check = (kk+(i_stride(1)-1)*(kk-1)):(kk+(i_stride(1)-1)*(kk-1)+k_vec(1)-1);
                check1 = any(row_check(:) == row(ii));
                check2 = any(col_check(:) == col_tmp);
                if check1 && check2
                    %out_dd_conf_from_in(jj,kk,ch_in,:) = out_dd_conf_from_in(jj,kk,ch_in,:) + 1;
                    %out_dd_conf_from_in(jj,kk,:) = 1;
                    out_dd_conf(jj,kk,:) = 1;
                end
            end
        end
    end
    
    %% conf from weight
    [row,col] = find(weight_td_conf);
    %in_dd_conf_from_weight_usage = zeros(i_vec);
    %out_dd_conf_from_weight_usage = zeros(o_vec);
    %in_dd_conf_from_weight = zeros(i_vec);
    %out_dd_conf_from_weight = zeros(o_vec);
    
    for ii = 1:size(row,1)
        
        ch_out = floor((col(ii)-1)/(k_vec(3)*k_vec(2)))+1;
        ch_in = mod(floor((col(ii)-1)/k_vec(2)),k_vec(3))+1;
        col_tmp = mod((col(ii)-1),k_vec(2))+1;
        
        % in dd
        row_start = row(ii);
        col_start = col_tmp;
        row_idx = row_start:i_stride(2):(row_start+(i_vec(2)-k_vec(2)));
        col_idx = col_start:i_stride(1):(col_start+(i_vec(1)-k_vec(1)));
        %in_dd_conf_from_weight_usage(row_idx,col_idx,:) = in_dd_conf_from_weight_usage(row_idx,col_idx,:) + 1;
        %in_dd_conf_from_weight(row_idx,col_idx,:) = 1;
        in_dd_conf(row_idx,col_idx,ch_in) = 1;
        
        % out dd
        %out_dd_conf_from_weight_usage(:,:,ch_out) = out_dd_conf_from_weight_usage(:,:,ch_out) + 1;
        %out_dd_conf_from_weight(:,:,ch_out) = 1;
        out_dd_conf(:,:,ch_out) = 1;
    end
    
    %% conf from output
    [row,col] = find(out_td_conf);
    %in_dd_conf_from_out_usage = zeros(i_vec);
    %weight_dd_conf_from_out_usage = zeros(k_vec);
    %in_dd_conf_from_out = zeros(i_vec);
    %weight_dd_conf_from_out = zeros(k_vec);
    
    for ii = 1:size(row,1)
        
        ch = mod(floor((col(ii)-1)/o_vec(2)),o_vec(3))+1;
        col_tmp = mod((col(ii)-1),o_vec(2))+1;
        
        % in dd
        row_start = 1 + (row(ii)-1)*i_stride(2);
        col_start = 1 + (col_tmp-1)*i_stride(1);
        row_idx = row_start:(row_start+k_vec(2)-1);
        col_idx = col_start:(col_start+k_vec(1)-1);
        %in_dd_conf_from_out_usage(row_idx,col_idx,:) = in_dd_conf_from_out_usage(row_idx,col_idx,:) + 1;
        %in_dd_conf_from_out(row_idx,col_idx,:) = 1;
        in_dd_conf(row_idx,col_idx,:) = 1;
        
        % weight dd
        %weight_dd_conf_from_out_usage(:,:,:,ch) = weight_dd_conf_from_out_usage(:,:,:,ch) + 1;
        %weight_dd_conf_from_out(:,:,:,ch) = 1;
        weight_dd_conf(:,:,:,ch) = 1;
        
    end
    
%     in_dd_conf_from_weight = in_dd_conf_from_weight_usage > 0;
%     in_dd_conf_from_out = in_dd_conf_from_out_usage > 0;
%     weight_dd_conf_from_in = weight_dd_conf_from_in_usage > 0;
%     weight_dd_conf_from_out = weight_dd_conf_from_out_usage > 0;
%     out_dd_conf_from_in = out_dd_conf_from_in_usage > 0;
%     out_dd_conf_from_weight = out_dd_conf_from_weight_usage > 0;
    
%     in_dd_conf = in_dd_conf_from_weight | in_dd_conf_from_out;
%     weight_dd_conf = weight_dd_conf_from_in | weight_dd_conf_from_out;
%     out_dd_conf = out_dd_conf_from_in | out_dd_conf_from_weight;
else
    assert(false,['Function GET_DD_CONF Error: Layer type "' layer_struct.type '" not supported!']);
end

res_vec = {in_dd_conf weight_dd_conf out_dd_conf};

end