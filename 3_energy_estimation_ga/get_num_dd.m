function res_vec = get_num_dd(num_td, layer_struct, string_dataflow)
%GET_NUM_DD
%
% (Usage)
%
% (Examples)
%
% (See also)

% $Author: Johnson Loh $  $Date: 2020/01/09 $ $Revision: 0.1 $
% Copyright: 

%% Check layer type
switch layer_struct.type
    case 'Linear'
        lt = 0;
    case 'Conv2d'
        lt = 1;
    otherwise
        assert(false,['Function GET_NUM_DD Error: Layer type "' layer_struct.type '" not supported!']);
end

%% Check Dataflow type
switch string_dataflow
    case 'input_stationary'
        dft = 0;
    case 'weight_stationary'
        dft = 1;
    case 'output_stationary'
        dft = 2;
    otherwise
        assert(false,['Function GET_NUM_DD Error: Dataflow type "' string_dataflow '" not supported!']);
end

%% Memory cost implementation
if lt == 0 && dft == 0
    % input data
    in_dd = 0;
    
    % weight data
    if num_td <= layer_struct.num_val_in
        weight_dd = num_td * layer_struct.num_val_out;
    else
        weight_dd = layer_struct.mem_paraminternal;
    end
    
    % output data
    out_dd = layer_struct.num_val_out;
elseif lt == 0 && dft == 1
    % input data
    if num_td <= layer_struct.num_val_in
        in_dd = num_td;
    else
        in_dd = layer_struct.num_val_in;
    end
    
    % weight data
    weight_dd = 0;
    
    % output data
    if num_td <= layer_struct.mem_paraminternal
        out_dd = ceil(num_td/layer_struct.num_val_in);
    else
        out_dd = layer_struct.num_val_out;
    end
    
elseif lt == 0 && dft == 2
    % input data
    if num_td == 0
        in_dd = 0;
    else
        in_dd = layer_struct.num_val_in;
    end
    
    % weight data
    if num_td <= layer_struct.num_val_out
        weight_dd = num_td * layer_struct.num_val_in;
    else
        weight_dd = layer_struct.mem_paraminternal;
    end    
    
    % output data
    out_dd = 0;
elseif lt == 1 && dft == 0
    k_vec = [layer_struct.dat(3:4) layer_struct.ch_in];
    i_vec = [(layer_struct.dim_in + 2*layer_struct.dat(7:8)) layer_struct.ch_in];
    o_vec = [layer_struct.dim_out layer_struct.ch_out];    
    i_stride = layer_struct.dat(5:6);
    
    % input data
    in_dd = 0;
    
    % weight data
    if num_td <= prod(i_vec)
        weight_dd = 0;
        for it = 0:(num_td-1)
            A = (mod(it,prod(i_vec(1:2))) < i_vec(1) * k_vec(2));
            B = (mod(it,i_vec(1)) < k_vec(1));
            if A && B
                psum = o_vec(3);
            else
                psum = 0;
            end
            weight_dd = weight_dd + psum;
        end
    else
        weight_dd = layer_struct.mem_paraminternal;
    end
    
    % output data
    if num_td <= prod(i_vec)
        if num_td > prod(i_vec(1:2))
            out_dd = layer_struct.num_val_out;
        else
            out_dd = 0;
            for it = 0:(num_td-1)
                A = (mod(it,prod(i_vec(1:2))) < i_vec(1) * o_vec(2));
                B = (mod(it,i_vec(1)) < o_vec(1));
                % stride conditions (asuming: stride < kernel size)
                C = (mod(mod(it,i_vec(1)), i_stride(1)) == 0);
                D = (mod(floor(it/i_vec(1)), i_stride(2)) == 0);
                if A && B && C && D
                    psum = o_vec(3);
                else
                    psum = 0;
                end
                out_dd = out_dd + psum;
            end
        end
    else
        out_dd = layer_struct.num_val_out;
    end
    
elseif lt == 1 && dft == 1
    k_vec = [layer_struct.dat(3:4) layer_struct.ch_in];
    i_vec = [(layer_struct.dim_in + 2*layer_struct.dat(7:8)) layer_struct.ch_in];
    o_vec = [layer_struct.dim_out layer_struct.ch_out];
    
    i_stride = layer_struct.dat(5:6);
    
    % input data
    if num_td <= layer_struct.mem_paraminternal
        if num_td > prod(k_vec)
            in_dd = prod(i_vec);
        else
            in_dd = 0;
            for it = 0:(num_td-1)
                tmp = mod(it,prod(k_vec(1:2)));
                % stride conditions (asuming: stride < kernel size)
                A = floor(tmp/(k_vec(1)*i_stride(2))) == 0;
                B = mod(tmp,k_vec(1)) < i_stride(1);
                if A && B
                    psum = ((i_vec(1)-k_vec(1))/i_stride(1)+1)*((i_vec(2)-k_vec(2))/i_stride(2)+1);
                elseif A && ~B
                    psum = ((i_vec(2)-k_vec(2))/i_stride(2)+1);
                elseif ~A && B
                    psum = ((i_vec(1)-k_vec(1))/i_stride(1)+1);
                else
                    psum = 1;
                end
                in_dd = in_dd + psum;
            end
        end
    else
        in_dd = prod(i_vec);
    end
    
    % weight data
    weight_dd = 0;
    
    % output data
    if num_td <= layer_struct.mem_paraminternal
        out_dd = ceil(num_td/prod(k_vec)) * prod(o_vec(1:2));
    else
        out_dd = layer_struct.num_val_out;
    end
elseif lt == 1 && dft == 2
    k_vec = [layer_struct.dat(3:4) layer_struct.ch_in];
    i_vec = [(layer_struct.dim_in + 2*layer_struct.dat(7:8)) layer_struct.ch_in];
    o_vec = [layer_struct.dim_out layer_struct.ch_out];
    
    i_stride = layer_struct.dat(5:6);
    
    % input data
    if num_td <= layer_struct.num_val_out
        if num_td > prod(o_vec(1:2))
            in_dd = prod(i_vec);
        else
            in_dd = 0;
            for it = 0:(num_td-1)
                tmp = mod(it,prod(o_vec(1:2)));
                A = floor(tmp/o_vec(1));
                B = mod(tmp,o_vec(1));
                if A == 0 && B == 0
                    psum = prod(k_vec);
                elseif A == 0 && B ~= 0
                    psum = k_vec(2) * k_vec(3) * i_stride(1);
                elseif A ~= 0 && B == 0
                    psum = k_vec(1) * k_vec(3) * i_stride(2);
                else
                    psum = k_vec(3) * prod(i_stride);
                end
                in_dd = in_dd + psum;
            end
        end
    else
        in_dd = prod(i_vec);
    end
    
    % weight data
    if num_td <= layer_struct.num_val_out
        weight_dd = ceil(num_td/prod(o_vec(1:2))) * prod(k_vec);
    else
        weight_dd = layer_struct.mem_paraminternal;
    end
    
    % output data
    out_dd = 0;
else
    assert(false,['Function GET_NUM_DD Error: Implementation with lt="' num2str(lt) '" and dft="' num2str(dft) '" does not exist!']);
end

res_vec = [in_dd weight_dd out_dd];

end
