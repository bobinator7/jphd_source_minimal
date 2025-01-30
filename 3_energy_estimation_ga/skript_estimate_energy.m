%% Energy Estimation for ANN
% $Author: Johnson Loh $  $Date: 2019/12/18 $ $Revision: 0.1 $
% Copyright: 
%
% TODO: create classes for hw_model, energy_microbenchmarks and net_arch

clear; clc; close all;

%% load data custom
% custom - alexnet def (from https://energyestimation.mit.edu/)
% custom_network_def_tab = readtable('NetworkConf_AlexNet_Unpruned.txt');
% custom_network_energy_tab = readtable('EnergyEstimation.txt');
% custom_net_spec = table2array(custom_network_def_tab);
% custom_net_energy = table2array(custom_network_energy_tab);
% custom_zero_entry_vec = [0.49;0.1710;0.1376;0.09643;0.1265;0.04873;0.03165;0.1875];
% custom_zero_entry_vec_extended = [0.49;0.49;0.49;0.49;0.1710;0.1710;0.1710;0.1376;0.1376;0.09643;0.09643;0.1265;0.1265;0.1265;0.04873;0.04873;0.04873;0.03165;0.03165;0.03165;0.1875];


% Example specs (from https://ieeexplore.ieee.org/document/7551407)
% 4-level memory hierarchy:
% DRAM -> global buffer -> PE FIFO -> register file
% values in bytes
% typical sizes for global buffer in 65nm: 100-300kB
%custom_mem_size_DRAM = inf;
%custom_mem_size_globbuff = 108 * 1024;     % 108kB total
%custom_mem_size_pfifo = nan;
%custom_mem_size_rf = 512 * 168;           % 168 PEs with 1kB register files each
%custom_mem_energy_DRAM = 200;
%custom_mem_energy_globbuff = 6;
%custom_mem_energy_pfifo = 2;
%custom_mem_energy_rf = 1;
%custom_area = 16000000; % 16mm^2
%custom_process = 65; % 65nm CMOS
%custom_precision = 16; % 16-bit fixed-point

custom_hw_model.type = 'custom';
custom_hw_model.mem_on_chip = (168 * 0.5 * 1024 + 108 * 1024) * 8; % convert bytes to bits
custom_hw_model.no_pe = 168;
custom_hw_model.precision = 16;    % bits
custom_hw_model.clock_rate = 200 * 10^6;
custom_hw_model.dataflow_strategy = 2;         % input stationary = 1; weight stationary = 2; output stationary = 3
custom_hw_model.mem_no_vals = custom_hw_model.mem_on_chip / custom_hw_model.precision;

%% memory models
% min model
min_hw_model.type = 'min';
min_hw_model.mem_on_chip = inf;
min_hw_model.no_pe = inf;
min_hw_model.precision = 16;
min_hw_model.dataflow_strategy = 0;

% max model
max_hw_model.type = 'max';
max_hw_model.mem_on_chip = 0;
max_hw_model.no_pe = 1;
max_hw_model.precision = 16;
max_hw_model.dataflow_strategy = 0;

% DRAM access max bandwidths
bitinterface = 64; % DIMM modules, RIMM 32/64, SIMM 8/32
max_bandwidth = 51.2; % GB/s PC4-51200 DDR5 SDRAM
dram_table = readtable('table_dram_speed.txt','Delimiter','\t','ReadVariableNames',false);

%% benchmark computation energy values
% placeholder values (normalized against 16-bit fixed-point MAC)
energy_microbenchmarks.op_mac = 1;
energy_microbenchmarks.op_compare = 1/17; %16-bit MAC -> 17 additions
energy_microbenchmarks.op_add = 1/17; %16-bit MAC -> 17 additions
energy_microbenchmarks.op_div = 1; % ~1 MAC
energy_microbenchmarks.mem_onchip_read = 6; % from custom worst case (global buffer)
energy_microbenchmarks.mem_onchip_write = 6; % from custom worst case (global buffer)
energy_microbenchmarks.mem_offchip_read = 200; % from custom
energy_microbenchmarks.mem_offchip_write = 200; % from custom

%% calc energy pytorch
% pytorch - alexnet def
net_arch = read_pytorch_named_modules('pytorch_netarch_formatted_v2.txt');

% compute architecture key parameters
net_arch = calc_key_param(net_arch);

% compute architecture computational costs
[res_comp,net_arch] = calc_comp_cost(net_arch, energy_microbenchmarks);

%% calc mem energy
[res_mem_min,net_arch_min] = calc_mem_cost(net_arch, energy_microbenchmarks, min_hw_model);
[res_mem_max,net_arch_max] = calc_mem_cost(net_arch, energy_microbenchmarks, max_hw_model);
[res_mem_custom,net_arch_custom] = calc_mem_cost(net_arch, energy_microbenchmarks, custom_hw_model);

%% plot stuff
e_model_min = [net_arch_min.e_comp_layer;net_arch_min.e_mem_layer]';
e_model_max = [net_arch_max.e_comp_layer;net_arch_max.e_mem_layer]';
e_model_custom = [net_arch_custom.e_comp_layer;net_arch_custom.e_mem_layer]';

% model min
figure
bar(1:21,e_model_min)
grid on
grid minor
set(gca,'xtick',1:21,'xticklabel',{net_arch.type})
xtickangle(90)
legend('comp','mem')
ylabel('E_{norm}')
xlabel('Layer')
title({'Minimum Energy Model (Weight fetch from on-chip memory for each iteration)' 'Alexnet v2, batch\_size = 128'})

% model max
figure
bar(1:21,e_model_max)
grid on
grid minor
set(gca,'xtick',1:21,'xticklabel',{net_arch.type})
xtickangle(90)
legend('comp','mem')
ylabel('E_{norm}')
xlabel('Layer')
title({'Maximum Energy Model' 'Alexnet v2, batch\_size = 128'})

% custom
% figure
% bar(1:21,e_model_custom)
% set(gca,'xtick',1:21,'xticklabel',{net_arch.type})
% xtickangle(90)
% legend('comp','mem')
% ylabel('E_{norm}')
% xlabel('Layer')
% title({'Custom Weightstationary Energy Model' 'Alexnet v2, batch\_size = 128'})

%%
custom_hw_model.dataflow_strategy = 1; 
[res_mem_custom1,net_arch_custom1] = calc_mem_cost(net_arch, energy_microbenchmarks, custom_hw_model);
custom_hw_model.dataflow_strategy = 2; 
[res_mem_custom2,net_arch_custom2] = calc_mem_cost(net_arch, energy_microbenchmarks, custom_hw_model);
custom_hw_model.dataflow_strategy = 3; 
[res_mem_custom3,net_arch_custom3] = calc_mem_cost(net_arch, energy_microbenchmarks, custom_hw_model);

[net_arch_custom1.e_mem_layer]
[net_arch_custom2.e_mem_layer]
[net_arch_custom3.e_mem_layer]

e_model_custom1 = [net_arch_custom1.e_comp_layer;net_arch_custom1.e_mem_layer]';
e_model_custom2 = [net_arch_custom2.e_comp_layer;net_arch_custom2.e_mem_layer]';
e_model_custom3 = [net_arch_custom3.e_comp_layer;net_arch_custom3.e_mem_layer]';

bar_data(:,1,1) = [net_arch_custom1.e_comp_layer]';
bar_data(:,1,2) = [net_arch_custom1.e_mem_layer]';
bar_data(:,2,1) = [net_arch_custom2.e_comp_layer]';
bar_data(:,2,2) = [net_arch_custom2.e_mem_layer]';
bar_data(:,3,1) = [net_arch_custom3.e_comp_layer]';
bar_data(:,3,2) = [net_arch_custom3.e_mem_layer]';

plotBarStackGroups(bar_data, {net_arch.type})
xtickangle(90)
grid on
grid minor
legend({'comp\_instat','mem\_instat','comp\_weightstat','mem\_weightstat','comp\_outstat','mem\_outstat'})

% e_model_custom1_scaled = [net_arch_custom1.e_comp_layer;net_arch_custom1.e_mem_layer]' .* custom_zero_entry_vec_extended;
% e_model_custom2_scaled = [net_arch_custom2.e_comp_layer;net_arch_custom2.e_mem_layer]' .* custom_zero_entry_vec_extended;
% e_model_custom3_scaled = [net_arch_custom3.e_comp_layer;net_arch_custom3.e_mem_layer]' .* custom_zero_entry_vec_extended;
% bar_data_scaled(:,1,1) = e_model_custom1_scaled(:,1);
% bar_data_scaled(:,1,2) = e_model_custom1_scaled(:,2);
% bar_data_scaled(:,2,1) = e_model_custom2_scaled(:,1);
% bar_data_scaled(:,2,2) = e_model_custom2_scaled(:,2);
% bar_data_scaled(:,3,1) = e_model_custom3_scaled(:,1);
% bar_data_scaled(:,3,2) = e_model_custom3_scaled(:,2);
% plotBarStackGroups(bar_data_scaled, {net_arch.type})
% xtickangle(90)
% grid on
% grid minor
% legend({'comp\_instat','mem\_instat','comp\_weightstat','mem\_weightstat','comp\_outstat','mem\_outstat'})

%% Extended Roofline Model
% Operations per cycle & Memory Access per cycle -> frequency independent
% evaluation of hw accelerator memory management/utilization

%figure

%% initial approach
% %layer = net_arch(2); %Conv Large
% %layer = net_arch(12); %Conv Small
% %layer = net_arch(16); %FC Large
% %layer = net_arch(21); %FC Small
% profile on
% for layer_no = [2 5 8 10 12 16 19 21]
%     in_dd = [];
%     ws_dd = [];
%     out_dd = [];
%     td_array = 1:10000;
%     for td = 1:10000
%         in_dd = cat(1,in_dd,get_num_dd(td, net_arch(layer_no), 'input_stationary'));
%         ws_dd = cat(1,ws_dd,get_num_dd(td, net_arch(layer_no), 'weight_stationary'));
%         out_dd = cat(1,out_dd,get_num_dd(td, net_arch(layer_no), 'output_stationary'));
%     end
%     %%
%     close all
%     figure
%     area(in_dd)
%     xlabel('No. PEs')
%     ylabel('No. Values in On-Chip memory')
%     legend('in\_dd','weight\_dd','out\_dd')
%     title({'Mem required for limited PEs',['Input\_stationary, ' net_arch(layer_no).name]})
%     print(['fig/' 'instat_layerno' num2str(layer_no)], '-dpng', '-r600')
%     %print(['fig' 'instat_layerno' num2str(layer_no)], '-depsc', '-r600')
%     savefig(['fig/' 'instat_layerno' num2str(layer_no)])
%     figure
%     area(ws_dd)
%     xlabel('No. PEs')
%     ylabel('No. Values in On-Chip memory')
%     legend('in\_dd','weight\_dd','out\_dd')
%     title({'Mem required for limited PEs',['Weight\_stationary, ' net_arch(layer_no).name]})
%     print(['fig/' 'weightstat_layerno' num2str(layer_no)], '-dpng', '-r600')
%     %print(['fig' 'weightstat_layerno' num2str(layer_no)], '-depsc', '-r600')
%     savefig(['fig/' 'weightstat_layerno' num2str(layer_no)])
%     figure
%     area(out_dd)
%     xlabel('No. PEs')
%     ylabel('No. Values in On-Chip memory')
%     legend('in\_dd','weight\_dd','out\_dd')
%     title({'Mem required for limited PEs',['Output\_stationary, ' net_arch(layer_no).name]})
%     print(['fig/' 'outstat_layerno' num2str(layer_no)], '-dpng', '-r600')
%     %print(['fig' 'outstat_layerno' num2str(layer_no)], '-depsc', '-r600')
%     savefig(['fig/' 'outstat_layerno' num2str(layer_no)])
% end
% profile viewer


%% Debug get_dd_conf
% %%example configuration for fc layer
% % td_conf1 = zeros(4096,1,'uint16');
% % td_conf1(1) = 1;
% % td_conf2 = zeros(4096,1000,'uint16');
% % td_conf2(1,1) = 1;
% % td_conf2(1,2) = 1;
% % td_conf2(2,2) = 1;
% % td_conf3 = zeros(1,1000,'uint16');
% % td_conf = {td_conf1 td_conf2 td_conf3};
% 
% %%example configuration for conv 11x11 stride 4 layer
% % td_conf_conv = {zeros(231,231,3) zeros(11,11,3,64) zeros(56,56,64)};
% % td_conf_conv{2}(122) = 1;
% % td_conf_conv{2}(134) = 1;
% % td_conf_conv{2}(630) = 1;
% % test = get_dd_conf(td_conf_conv, net_arch(2));
% 
% %%example configuration for conv 3x3 stride 1 layer
% %input stationary
% td_conf_conv = {zeros(15,15,256) zeros(3,3,256,256) zeros(13,13,256)};
% td_conf_conv{1}(226) = 1;
% td_conf_conv{1}(242) = 1;
% td_conf_conv{1}(483) = 1;
% % td_conf_conv{2}(10) = 1;
% % td_conf_conv{2}(14) = 1;
% % td_conf_conv{2}(2331) = 1;
% % td_conf_conv{3}(170) = 1;
% % td_conf_conv{3}(184) = 1;
% % td_conf_conv{3}(367) = 1;
% test = get_dd_conf(td_conf_conv, net_arch(12));
% 
% figure
% sgtitle('Input stationary')
% subplot(3,3,1), imshow(td_conf_conv{1}(:,:,1)), ...
%     grid on, xticks(0.5:1:(size(td_conf_conv{1},1)+0.5)), xticklabels([]), yticks(0.5:1:(size(td_conf_conv{1},2)+0.5)), yticklabels([]), ...
%     axis on, title('In TD, ch\_in=1')
% subplot(3,3,2), imshow(td_conf_conv{1}(:,:,2)), ...
%     grid on, xticks(0.5:1:(size(td_conf_conv{1},1)+0.5)), xticklabels([]), yticks(0.5:1:(size(td_conf_conv{1},2)+0.5)), yticklabels([]), ...
%     axis on, title('In TD, ch\_in=2')
% subplot(3,3,3), imshow(td_conf_conv{1}(:,:,3)), ...
%     grid on, xticks(0.5:1:(size(td_conf_conv{1},1)+0.5)), xticklabels([]), yticks(0.5:1:(size(td_conf_conv{1},2)+0.5)), yticklabels([]), ...
%     axis on, title('In TD, ch\_in=3')
% subplot(3,3,4), imshow(test{2}(:,:,1,1)), ...
%     grid on, xticks(0.5:1:(size(test{2},1)+0.5)), xticklabels([]), yticks(0.5:1:(size(test{2},2)+0.5)), yticklabels([]), ...
%     axis on, title({'Weight DD' 'ch\_in=1, ch\_out=1'})
% subplot(3,3,5), imshow(test{2}(:,:,2,1)), ...
%     grid on, xticks(0.5:1:(size(test{2},1)+0.5)), xticklabels([]), yticks(0.5:1:(size(test{2},2)+0.5)), yticklabels([]), ...
%     axis on, title({'Weight DD' 'ch\_in=2, ch\_out=1'})
% subplot(3,3,6), imshow(test{2}(:,:,3,1)), ...
%     grid on, xticks(0.5:1:(size(test{2},1)+0.5)), xticklabels([]), yticks(0.5:1:(size(test{2},2)+0.5)), yticklabels([]), ...
%     axis on, title({'Weight DD' 'ch\_in=3, ch\_out=2'})
% subplot(3,3,7), imshow(test{3}(:,:,1)), ...
%     grid on, xticks(0.5:1:(size(test{3},1)+0.5)), xticklabels([]), yticks(0.5:1:(size(test{3},2)+0.5)), yticklabels([]), ...
%     axis on, title('Out DD, ch\_out=1')
% subplot(3,3,8), imshow(test{3}(:,:,2)), ...
%     grid on, xticks(0.5:1:(size(test{3},1)+0.5)), xticklabels([]), yticks(0.5:1:(size(test{3},2)+0.5)), yticklabels([]), ...
%     axis on, title('Out DD, ch\_out=2')
% subplot(3,3,9), imshow(test{3}(:,:,3)), ...
%     grid on, xticks(0.5:1:(size(test{3},1)+0.5)), xticklabels([]), yticks(0.5:1:(size(test{3},2)+0.5)), yticklabels([]), ...
%     axis on, title('Out DD, ch\_out=3')
% 
% %weight stationary
% td_conf_conv = {zeros(15,15,256) zeros(3,3,256,256) zeros(13,13,256)};
% % td_conf_conv{1}(226) = 1;
% % td_conf_conv{1}(242) = 1;
% % td_conf_conv{1}(483) = 1;
% td_conf_conv{2}(10) = 1;
% td_conf_conv{2}(14) = 1;
% td_conf_conv{2}(2331) = 1;
% % td_conf_conv{3}(170) = 1;
% % td_conf_conv{3}(184) = 1;
% % td_conf_conv{3}(367) = 1;
% test = get_dd_conf(td_conf_conv, net_arch(12));
% 
% figure
% sgtitle('Weight stationary')
% subplot(3,3,1), imshow(test{1}(:,:,1)), ...
%     grid on, xticks(0.5:1:(size(test{1},1)+0.5)), xticklabels([]), yticks(0.5:1:(size(test{1},2)+0.5)), yticklabels([]), ...
%     axis on, title('In DD, ch\_in=1')
% subplot(3,3,2), imshow(test{1}(:,:,2)), ...
%     grid on, xticks(0.5:1:(size(test{1},1)+0.5)), xticklabels([]), yticks(0.5:1:(size(test{1},2)+0.5)), yticklabels([]), ...
%     axis on, title('In DD, ch\_in=2')
% subplot(3,3,3), imshow(test{1}(:,:,3)), ...
%     grid on, xticks(0.5:1:(size(test{1},1)+0.5)), xticklabels([]), yticks(0.5:1:(size(test{1},2)+0.5)), yticklabels([]), ...
%     axis on, title('In DD, ch\_in=3')
% subplot(3,3,4), imshow(td_conf_conv{2}(:,:,1,1)), ...
%     grid on, xticks(0.5:1:(size(td_conf_conv{2},1)+0.5)), xticklabels([]), yticks(0.5:1:(size(td_conf_conv{2},2)+0.5)), yticklabels([]), ...
%     axis on, title({'Weight TD' 'ch\_in=1, ch\_out=1'})
% subplot(3,3,5), imshow(td_conf_conv{2}(:,:,2,1)), ...
%     grid on, xticks(0.5:1:(size(td_conf_conv{2},1)+0.5)), xticklabels([]), yticks(0.5:1:(size(td_conf_conv{2},2)+0.5)), yticklabels([]), ...
%     axis on, title({'Weight TD' 'ch\_in=2, ch\_out=1'})
% subplot(3,3,6), imshow(td_conf_conv{2}(:,:,3,2)), ...
%     grid on, xticks(0.5:1:(size(td_conf_conv{2},1)+0.5)), xticklabels([]), yticks(0.5:1:(size(td_conf_conv{2},2)+0.5)), yticklabels([]), ...
%     axis on, title({'Weight TD' 'ch\_in=3, ch\_out=2'})
% subplot(3,3,7), imshow(test{3}(:,:,1)), ...
%     grid on, xticks(0.5:1:(size(test{3},1)+0.5)), xticklabels([]), yticks(0.5:1:(size(test{3},2)+0.5)), yticklabels([]), ...
%     axis on, title('Out DD, ch\_out=1')
% subplot(3,3,8), imshow(test{3}(:,:,2)), ...
%     grid on, xticks(0.5:1:(size(test{3},1)+0.5)), xticklabels([]), yticks(0.5:1:(size(test{3},2)+0.5)), yticklabels([]), ...
%     axis on, title('Out DD, ch\_out=2')
% subplot(3,3,9), imshow(test{3}(:,:,3)), ...
%     grid on, xticks(0.5:1:(size(test{3},1)+0.5)), xticklabels([]), yticks(0.5:1:(size(test{3},2)+0.5)), yticklabels([]), ...
%     axis on, title('Out DD, ch\_out=3')
% 
% %output stationary
% td_conf_conv = {zeros(15,15,256) zeros(3,3,256,256) zeros(13,13,256)};
% % td_conf_conv{1}(226) = 1;
% % td_conf_conv{1}(242) = 1;
% % td_conf_conv{1}(483) = 1;
% % td_conf_conv{2}(10) = 1;
% % td_conf_conv{2}(14) = 1;
% % td_conf_conv{2}(2331) = 1;
% td_conf_conv{3}(170) = 1;
% td_conf_conv{3}(184) = 1;
% td_conf_conv{3}(367) = 1;
% test = get_dd_conf(td_conf_conv, net_arch(12));
% 
% figure
% sgtitle('Output stationary')
% subplot(3,3,1), imshow(test{1}(:,:,1)), ...
%     grid on, xticks(0.5:1:(size(test{1},1)+0.5)), xticklabels([]), yticks(0.5:1:(size(test{1},2)+0.5)), yticklabels([]), ...
%     axis on, title('In DD, ch\_in=1')
% subplot(3,3,2), imshow(test{1}(:,:,2)), ...
%     grid on, xticks(0.5:1:(size(test{1},1)+0.5)), xticklabels([]), yticks(0.5:1:(size(test{1},2)+0.5)), yticklabels([]), ...
%     axis on, title('In DD, ch\_in=2')
% subplot(3,3,3), imshow(test{1}(:,:,3)), ...
%     grid on, xticks(0.5:1:(size(test{1},1)+0.5)), xticklabels([]), yticks(0.5:1:(size(test{1},2)+0.5)), yticklabels([]), ...
%     axis on, title('In DD, ch\_in=3')
% subplot(3,3,4), imshow(test{2}(:,:,1,1)), ...
%     grid on, xticks(0.5:1:(size(test{2},1)+0.5)), xticklabels([]), yticks(0.5:1:(size(test{2},2)+0.5)), yticklabels([]), ...
%     axis on, title({'Weight DD' 'ch\_in=1, ch\_out=1'})
% subplot(3,3,5), imshow(test{2}(:,:,2,1)), ...
%     grid on, xticks(0.5:1:(size(test{2},1)+0.5)), xticklabels([]), yticks(0.5:1:(size(test{2},2)+0.5)), yticklabels([]), ...
%     axis on, title({'Weight DD' 'ch\_in=2, ch\_out=1'})
% subplot(3,3,6), imshow(test{2}(:,:,3,2)), ...
%     grid on, xticks(0.5:1:(size(test{2},1)+0.5)), xticklabels([]), yticks(0.5:1:(size(test{2},2)+0.5)), yticklabels([]), ...
%     axis on, title({'Weight DD' 'ch\_in=3, ch\_out=2'})
% subplot(3,3,7), imshow(td_conf_conv{3}(:,:,1)), ...
%     grid on, xticks(0.5:1:(size(td_conf_conv{3},1)+0.5)), xticklabels([]), yticks(0.5:1:(size(td_conf_conv{3},2)+0.5)), yticklabels([]), ...
%     axis on, title('Out TD, ch\_out=1')
% subplot(3,3,8), imshow(td_conf_conv{3}(:,:,2)), ...
%     grid on, xticks(0.5:1:(size(td_conf_conv{3},1)+0.5)), xticklabels([]), yticks(0.5:1:(size(td_conf_conv{3},2)+0.5)), yticklabels([]), ...
%     axis on, title('Out TD, ch\_out=2')
% subplot(3,3,9), imshow(td_conf_conv{3}(:,:,3)), ...
%     grid on, xticks(0.5:1:(size(td_conf_conv{3},1)+0.5)), xticklabels([]), yticks(0.5:1:(size(td_conf_conv{3},2)+0.5)), yticklabels([]), ...
%     axis on, title('Out TD, ch\_out=3')


%% Debug optim_get_seq_cost
% layer_struct = net_arch(2);
% num_seq_elem_input = 231*231*3;
% num_seq_elem_weight = 11*11*3*64;
% num_seq_elem_output = 56*56*64;
% no_pe = 64;
% 
% % naive (ref)
% disp('Reference Scheduling - Naive (Sequential Processing)')
% tic
% cost_ref_instat = optim_get_seq_cost(no_pe, {1:num_seq_elem_input}, 1, layer_struct);
% toc
% tic
% cost_ref_weightstat = optim_get_seq_cost(no_pe, {1:num_seq_elem_weight}, 2, layer_struct);
% toc
% tic
% cost_ref_outstat = optim_get_seq_cost(no_pe, {1:num_seq_elem_output}, 3, layer_struct);
% toc
% 
% % random (ref)
% disp('Reference Scheduling - Random')
% tic
% cost_rand_instat = optim_get_seq_cost(no_pe, {randperm(num_seq_elem_input)}, 1, layer_struct);
% toc
% tic
% cost_rand_weightstat = optim_get_seq_cost(no_pe, {randperm(num_seq_elem_weight)}, 2, layer_struct);
% toc
% tic
% cost_rand_outstat = optim_get_seq_cost(no_pe, {randperm(num_seq_elem_output)}, 3, layer_struct);
% toc
% 
% % optimized
% FitnessFcn = @(x)optim_get_seq_cost_par(no_pe,x,2,layer_struct);
% options = optimoptions(@ga, 'PopulationType', 'custom','InitialPopulationRange', ...
%                             [1;num_seq_elem_weight]);
% options = optimoptions(options,'CreationFcn',@optim_create_permutations, ...
%                         'CrossoverFcn',@optim_crossover_permutation, ...
%                         'MutationFcn',@optim_mutate_permutation, ...
%                         'MaxGenerations',2000,'PopulationSize',100, ...
%                         'MaxStallGenerations',100,'UseVectorized',false, ...
%                         'UseParallel',true);
% 
% numberOfVariables = num_seq_elem_weight;
% 
% parpool('local');
% tic
% [x,fval,reason,output] = ...
%     ga(FitnessFcn,numberOfVariables,[],[],[],[],[],[],[],options);
% toc
% delete(gcp)
