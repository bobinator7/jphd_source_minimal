%% Visualization of the state of memory needed to process one set of independent variables in a ANN layer
% $Author: Johnson Loh $  $Date: 2019/12/18 $ $Revision: 0.1 $
% Copyright: 
clear; clc; close all;

%% Visualize variable space
tmp.name = 'testlayer';
tmp.type = 'Conv2d';
tmp.dat = [2 2 3 3 2 2 0 0];
tmp.dim_in = [7 7];
tmp.ch_in = 2; tmp.dim_out = [3 3]; tmp.ch_out = 2;
tmp.num_val_in = prod([(tmp.dim_in + 2*tmp.dat(7:8)) tmp.ch_in]);%prod(tmp.dim_in) * tmp.ch_in;
tmp.num_val_out = prod(tmp.dim_out) * tmp.ch_out;
tmp.num_val_param_per_out = prod(tmp.dat(3:4)) * tmp.ch_in;
tmp.mem_paraminternal = prod(tmp.dat(3:4)) * tmp.ch_in * tmp.ch_out;

td_conf_conv = {zeros(7,7,2) zeros(3,3,2,2) zeros(3,3,2)};
for ii = 1:numel(td_conf_conv{2})
td_conf_conv = {zeros(7,7,2) zeros(3,3,2,2) zeros(3,3,2)};
td_conf_conv{2}(1:18) = 1;
dd_conf_conv = get_dd_conf(td_conf_conv, tmp);

figure
subplot(3,4,[1,2]), imshow(dd_conf_conv{1}(:,:,1)), ...
    grid on, xticks(0.5:1:(size(td_conf_conv{1},1)+0.5)), xticklabels([]), yticks(0.5:1:(size(td_conf_conv{1},2)+0.5)), yticklabels([]), ...
    axis on
subplot(3,4,[3,4]), imshow(dd_conf_conv{1}(:,:,2)), ...
    grid on, xticks(0.5:1:(size(td_conf_conv{1},1)+0.5)), xticklabels([]), yticks(0.5:1:(size(td_conf_conv{1},2)+0.5)), yticklabels([]), ...
    axis on
subplot(3,4,5), imshow(td_conf_conv{2}(:,:,1,1)), ...
    grid on, xticks(0.5:1:(size(td_conf_conv{2},1)+0.5)), xticklabels([]), yticks(0.5:1:(size(td_conf_conv{2},2)+0.5)), yticklabels([]), ...
    axis on
subplot(3,4,6), imshow(td_conf_conv{2}(:,:,1,2)), ...
    grid on, xticks(0.5:1:(size(td_conf_conv{2},1)+0.5)), xticklabels([]), yticks(0.5:1:(size(td_conf_conv{2},2)+0.5)), yticklabels([]), ...
    axis on
subplot(3,4,7), imshow(td_conf_conv{2}(:,:,2,1)), ...
    grid on, xticks(0.5:1:(size(td_conf_conv{2},1)+0.5)), xticklabels([]), yticks(0.5:1:(size(td_conf_conv{2},2)+0.5)), yticklabels([]), ...
    axis on
subplot(3,4,8), imshow(td_conf_conv{2}(:,:,2,2)), ...
    grid on, xticks(0.5:1:(size(td_conf_conv{2},1)+0.5)), xticklabels([]), yticks(0.5:1:(size(td_conf_conv{2},2)+0.5)), yticklabels([]), ...
    axis on
subplot(3,4,[9,10]), imshow(dd_conf_conv{3}(:,:,1)), ...
    grid on, xticks(0.5:1:(size(td_conf_conv{3},1)+0.5)), xticklabels([]), yticks(0.5:1:(size(td_conf_conv{3},2)+0.5)), yticklabels([]), ...
    axis on
subplot(3,4,[11,12]), imshow(dd_conf_conv{3}(:,:,2)), ...
    grid on, xticks(0.5:1:(size(td_conf_conv{3},1)+0.5)), xticklabels([]), yticks(0.5:1:(size(td_conf_conv{3},2)+0.5)), yticklabels([]), ...
    axis on

annotation('arrow',[0.3 0.2],[0.7 0.63],'color','r','linewidth',2);
annotation('arrow',[0.35 0.42],[0.7 0.63],'color','r','linewidth',2);
annotation('arrow',[1-0.3 1-0.2],[0.7 0.63],'color','r','linewidth',2);
annotation('arrow',[1-0.35 1-0.42],[0.7 0.63],'color','r','linewidth',2);

annotation('arrow',[0.2 0.3],[0.4 0.33],'color','r','linewidth',2);
annotation('arrow',[0.6 0.35],[0.4 0.33],'color','r','linewidth',2);
annotation('arrow',[1-0.2 1-0.3],[0.4 0.33],'color','r','linewidth',2);
annotation('arrow',[1-0.6 1-0.35],[0.4 0.33],'color','r','linewidth',2);

% annotation('rectangle',[0.1 0.7 0.84 0.225],'Color','green','linewidth',2)
% annotation('textbox', [0, 0.8, 0, 0], 'string', 'Data in PEs','Color','green')
annotation('rectangle',[0.1 0.4 0.84 0.225],'Color','green','linewidth',2)
annotation('textbox', [0, 0.5, 0, 0], 'string', 'Data in PEs','Color','green')
% annotation('rectangle',[0.1 0.1 0.84 0.225],'Color','green','linewidth',2)
% annotation('textbox', [0, 0.2, 0, 0], 'string', 'Data in PEs','Color','green')

sgtitle('Weight stationary, Stride 2x2')
print(['gif/' 'weightstat' num2str(ii,'%03d')], '-dpng')
close all;
end

%% create video
imageNames = dir(fullfile('gif','*.png'));
imageNames = {imageNames.name}';
outputVideo = VideoWriter(fullfile('gif','out'), 'MPEG-4');
outputVideo.FrameRate = 4;
open(outputVideo)
for ii = 1:length(imageNames)
   img = imread(fullfile('gif',imageNames{ii}));
   writeVideo(outputVideo,img)
end
close(outputVideo)