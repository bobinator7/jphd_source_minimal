function model_arch_struct = read_pytorch_named_modules(filename)
%READ_PYTORCH_NAMED_MODULES
%
% (Usage)
%
% (Examples)
%
% (See also)

% $Author: Johnson Loh $  $Date: 2019/12/18 $ $Revision: 0.1 $
% Copyright: 

% open file
fid = fopen(filename,'r');

layer_no = 1;
tline = fgetl(fid);
while ischar(tline)
    % read line info and data
    line_entries = split(tline);
    arch(layer_no).name = line_entries{1};
    arch(layer_no).type = line_entries{2};
    arch(layer_no).dat = cellfun(@str2num,line_entries(3:end)).';
    layer_no = layer_no + 1;
    tline = fgetl(fid);
end

% close file
fclose(fid);
model_arch_struct = arch;
end