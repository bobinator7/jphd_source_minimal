function [outputArg1,outputArg2] = plot_seq_weight(seq,tmp)
%PLOT_SEQ Summary of this function goes here
%
% (Usage)
%
% (Examples)
%
% (See also) https://www.mathworks.com/help/gads/custom-data-type-optimization-using-ga.html

% $Author: Johnson Loh $  $Date: 2020/01/28 $ $Revision: 0.1 $
% Copyright: 

td_conf_conv = {zeros(7,7,2) zeros(3,3,2,2) zeros(5,5,2)};
cnt=1;
for ii = seq{1}
td_conf_conv{2}(:) = 0;
td_conf_conv{2}(ii) = 1;
dd_conf_conv = get_dd_conf(td_conf_conv, tmp);

figure
sfh1 = subplot(3,4,[1,2]), imshow(dd_conf_conv{1}(:,:,1)), ...
    grid on, xticks(0.5:1:(size(td_conf_conv{1},1)+0.5)), xticklabels([]), yticks(0.5:1:(size(td_conf_conv{1},2)+0.5)), yticklabels([]), ...
    axis on
sfh2 = subplot(3,4,[3,4]), imshow(dd_conf_conv{1}(:,:,2)), ...
    grid on, xticks(0.5:1:(size(td_conf_conv{1},1)+0.5)), xticklabels([]), yticks(0.5:1:(size(td_conf_conv{1},2)+0.5)), yticklabels([]), ...
    axis on
sfh3 = subplot(3,4,5), imshow(td_conf_conv{2}(:,:,1,1)), ...
    grid on, xticks(0.5:1:(size(td_conf_conv{2},1)+0.5)), xticklabels([]), yticks(0.5:1:(size(td_conf_conv{2},2)+0.5)), yticklabels([]), ...
    axis on
sfh4 = subplot(3,4,6), imshow(td_conf_conv{2}(:,:,1,2)), ...
    grid on, xticks(0.5:1:(size(td_conf_conv{2},1)+0.5)), xticklabels([]), yticks(0.5:1:(size(td_conf_conv{2},2)+0.5)), yticklabels([]), ...
    axis on
sfh5 = subplot(3,4,7), imshow(td_conf_conv{2}(:,:,2,1)), ...
    grid on, xticks(0.5:1:(size(td_conf_conv{2},1)+0.5)), xticklabels([]), yticks(0.5:1:(size(td_conf_conv{2},2)+0.5)), yticklabels([]), ...
    axis on
sfh6 = subplot(3,4,8), imshow(td_conf_conv{2}(:,:,2,2)), ...
    grid on, xticks(0.5:1:(size(td_conf_conv{2},1)+0.5)), xticklabels([]), yticks(0.5:1:(size(td_conf_conv{2},2)+0.5)), yticklabels([]), ...
    axis on
sfh7 = subplot(3,4,[9,10]), imshow(dd_conf_conv{3}(:,:,1)), ...
    grid on, xticks(0.5:1:(size(td_conf_conv{3},1)+0.5)), xticklabels([]), yticks(0.5:1:(size(td_conf_conv{3},2)+0.5)), yticklabels([]), ...
    axis on
sfh8 = subplot(3,4,[11,12]), imshow(dd_conf_conv{3}(:,:,2)), ...
    grid on, xticks(0.5:1:(size(td_conf_conv{3},1)+0.5)), xticklabels([]), yticks(0.5:1:(size(td_conf_conv{3},2)+0.5)), yticklabels([]), ...
    axis on
set(sfh1, 'GridColor', 'r')
sfh3.Position = sfh3.Position .* [sfh3.Position(3)/2 sfh3.Position(4)/2 0 0] + sfh3.Position .* [1 1 0.5 0.5];
sfh4.Position = sfh4.Position .* [sfh4.Position(3)/2 sfh4.Position(4)/2 0 0] + sfh4.Position .* [1 1 0.5 0.5];
sfh5.Position = sfh5.Position .* [sfh5.Position(3)/2 sfh5.Position(4)/2 0 0] + sfh5.Position .* [1 1 0.5 0.5];
sfh6.Position = sfh6.Position .* [sfh6.Position(3)/2 sfh6.Position(4)/2 0 0] + sfh6.Position .* [1 1 0.5 0.5];

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

print(['gif/' 'weightstat' num2str(cnt,'%03d')], '-dpng')
close all;
cnt=cnt+1;
end

% create video
imageNames = dir(fullfile('gif','weightstat*.png'));
imageNames = {imageNames.name}';
outputVideo = VideoWriter(fullfile('gif','weightstat'), 'MPEG-4');
outputVideo.FrameRate = 4;
open(outputVideo)
for ii = 1:length(imageNames)
   img = imread(fullfile('gif',imageNames{ii}));
   writeVideo(outputVideo,img)
end
close(outputVideo)

end

