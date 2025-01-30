function [no_pe_out, maps_per_tile] = mem_conv_floor_pe(n_pe, seq_type, layer_struct)
%MEM_CONV_FLOOR_PE
%
% (Usage)
%
% (Examples)
%
% (See also)

% $Author: Johnson Loh $  $Date: 2019/02/12 $ $Revision: 0.1 $
% Copyright: 

%% 
switch seq_type
    case 1
        map_size = prod(layer_struct.dim_in);
    case 2
        map_size = prod(layer_struct.dat(3:4));
    case 3
        map_size = prod(layer_struct.dim_out);
    otherwise
        assert(false,'Function GET_SEQ_COST Error: Sequence type not defined!');
end

maps_per_tile = floor(n_pe/map_size);
no_pe_out = maps_per_tile * map_size;

end