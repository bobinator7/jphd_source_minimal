function [onchip_read_accesses, onchip_write_accesses, read_accesses,write_accesses] = mem_conv_accesses(no_pe, seq_type, layer_struct)
%MEM_CONV_OFFCHIP_ACCESSES
%
% (Usage)
%
% (Examples)
%
% (See also)

% $Author: Johnson Loh $  $Date: 2019/02/12 $ $Revision: 0.1 $
% Copyright: 

%% check if what tiling is appropriate (tile channel only <-> addtl. dimension for in map tiling)
in_inc = prod(layer_struct.dim_in  + 2*layer_struct.dat(7:8));
weight_inc = prod(layer_struct.dat(3:4));
out_inc = prod(layer_struct.dim_out);
switch seq_type
    case 1
        map_size = in_inc;
    case 2
        map_size = weight_inc;
    case 3
        map_size = out_inc;
    otherwise
        assert(false,'Function MEM_CONV_OFFCHIP_ACCESSES Error: Sequence type not defined!');
end

% Assumption: On-chip memory is at least as large as the requirements of
% the pe array
if no_pe >= map_size
    % pe can keep at least one map and required on-chip memory is larger than one iteration
    tiling_type = 1;
else
    tiling_type = 0;
end

%% params
% pe tile size and no maps per tile
if tiling_type == 1
    [no_pe_out, no_maps_per_tile] = mem_conv_floor_pe(no_pe, seq_type, layer_struct);
else
    no_pe_out = map_size;
    no_maps_per_tile = nan;
end

% no tiles per layer
switch seq_type
    case 1
        tiles_per_layer = layer_struct.num_val_in / no_pe_out;
    case 2
        tiles_per_layer = layer_struct.mem_paraminternal / no_pe_out;
    case 3
        tiles_per_layer = layer_struct.num_val_out / no_pe_out;
    otherwise
        assert(false,'Function MEM_CONV_OFFCHIP_ACCESSES Error: Sequence type not defined!');
end

%% accesses per tile (assuming reuse only over consecutive iterations)
if tiling_type == 1
    switch seq_type
        case 1
            % off-chip
            read_accesses_per_tile_dd = weight_inc * no_maps_per_tile * layer_struct.ch_out + out_inc * layer_struct.ch_out;
            write_accesses_per_tile_dd = out_inc * layer_struct.ch_out;
            read_accesses_per_tile_td = no_pe_out;
            write_accesses_per_tile_td = 0;
            
            % on-chip
            onchip_in_read = no_pe_out;
            onchip_weight_read = (weight_inc * out_inc) * no_maps_per_tile * layer_struct.ch_out;
            onchip_out_read = out_inc * layer_struct.ch_out;
            onchip_read = onchip_in_read + onchip_weight_read + onchip_out_read;
            onchip_in_write = 0;
            onchip_weight_write = 0;
            onchip_out_write = out_inc * layer_struct.ch_out;
            onchip_write = onchip_in_write + onchip_weight_write + onchip_out_write;
        case 2
            % off-chip
            no_in = ceil(no_maps_per_tile/layer_struct.ch_out);
            no_out = min(no_maps_per_tile, layer_struct.ch_out);
            read_accesses_per_tile_dd = in_inc * no_in + out_inc * no_out;
            write_accesses_per_tile_dd = out_inc * no_out;
            read_accesses_per_tile_td = no_pe_out;
            write_accesses_per_tile_td = 0;
            
            % on-chip
            onchip_in_read = (weight_inc * out_inc) * no_in;
            onchip_weight_read = no_pe_out;
            onchip_out_read = out_inc * no_out;
            onchip_read = onchip_in_read + onchip_weight_read + onchip_out_read;
            onchip_in_write = 0;
            onchip_weight_write = 0;
            onchip_out_write = out_inc * no_out;
            onchip_write = onchip_in_write + onchip_weight_write + onchip_out_write;
        case 3
            % off-chip
            read_accesses_per_tile_dd = weight_inc * no_maps_per_tile * layer_struct.ch_in + in_inc * layer_struct.ch_in;
            write_accesses_per_tile_dd = 0;
            read_accesses_per_tile_td = no_pe_out;
            write_accesses_per_tile_td = no_pe_out;
            
            % on-chip
            onchip_in_read = (weight_inc * out_inc) * no_maps_per_tile * layer_struct.ch_out;
            onchip_weight_read = (weight_inc * out_inc) * no_maps_per_tile * layer_struct.ch_out;
            onchip_out_read = no_pe_out;
            onchip_read = onchip_in_read + onchip_weight_read + onchip_out_read;
            onchip_in_write = 0;
            onchip_weight_write = 0;
            onchip_out_write = no_pe_out;
            onchip_write = onchip_in_write + onchip_weight_write + onchip_out_write;
        otherwise
            assert(false,'Function MEM_CONV_OFFCHIP_ACCESSES Error: Sequence type not defined!');
    end
    read_accesses_per_tile = read_accesses_per_tile_td + read_accesses_per_tile_dd;
    write_accesses_per_tile = write_accesses_per_tile_td + write_accesses_per_tile_dd;
else
    [onchip_read, onchip_write, read_accesses_per_tile, write_accesses_per_tile] = optim_get_reducedseq_cost(no_pe, seq_type, layer_struct);
end

%%
read_accesses = read_accesses_per_tile * tiles_per_layer;
write_accesses = write_accesses_per_tile * tiles_per_layer;

%%
onchip_read_accesses = onchip_read * tiles_per_layer;
onchip_write_accesses = onchip_write * tiles_per_layer;

end