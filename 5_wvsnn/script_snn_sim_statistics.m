%% Author: Johnson Loh; Date: 01.03.2022
clear all; close all; clc; fclose('all');


%% get nn data
% maxpool after each conv
% dimensions: 4D -> out channels, in channels, dim1, dim2 (2D data)
% 3D -> out channels, in channels, dim1 (1D data)
% FC -> 70 channels * 11 input samples (max 3 pooled)
% Effective window is 16816 
% -> ((((((11*3)+4)*3)+4)*3)+4)*3)+4) *2^4 (FC 11 samples, pool 3, cnn kernel 5)
% Step for new fc sample is 1296 (4.32 sec)
% -> Pool outputs data for every 3rd input sample, DWT outputs data for
% every 2nd input sample
load('model_state_dict4_snn.mat')

%% DWT (full precision)
% DWT filter info
wname='db2';
[LoD,HiD,LoR,HiR] = wfilters(wname);

%% Quantization param + input
% fixed point type
ntFP = numerictype(1,14,12);
quant_max_val = (1-2^-ntFP.FractionLength);
ntFP_dwt = numerictype(1,16,12);
ntFP_param = numerictype(1,16,12);

% quantized weights biases
w_tmp = fliplr(permute(dict.conv1_weight,[3,4,1,2]));
w_c1 = quantPrintNReturn(w_tmp, ntFP_param, '');
b_c1 = quantPrintNReturn(dict.conv1_bias, ntFP_param, '');

w_tmp = fliplr(permute(dict.conv2_weight,[2,3,1]));
w_c2 = quantPrintNReturn(w_tmp, ntFP_param, '');
b_c2 = quantPrintNReturn(dict.conv2_bias, ntFP_param, '');

w_tmp = fliplr(permute(dict.conv3_weight,[2,3,1]));
w_c3 = quantPrintNReturn(w_tmp, ntFP_param, '');
b_c3 = quantPrintNReturn(dict.conv3_bias, ntFP_param, '');

w_tmp = fliplr(permute(dict.conv4_weight,[2,3,1]));
w_c4 = quantPrintNReturn(w_tmp, ntFP_param, '');
b_c4 = quantPrintNReturn(dict.conv4_bias, ntFP_param, '');

%w_tmp = permute(reshape(dict.fc_weight',11,[],4), [2 1 3]);
w_fc = quantPrintNReturn(dict.fc_weight, ntFP_param, '');
b_fc = quantPrintNReturn(dict.fc_bias, ntFP_param, '');
%w_fc_flat = reshape(permute(w_fc,[2,1,3]),1,[],4);

%% prune network
% select channels, where at least one weight in output channel is not
% quantized to zero
nz_ch_conv1 = sum(w_c1~=0,[1,2])~=0;
nz_ch_conv2 = sum(w_c2~=0,[1,2])~=0;
nz_ch_conv3 = sum(w_c3~=0,[1,2])~=0;
nz_ch_conv4 = sum(w_c4~=0,[1,2])~=0;
nz_ch_fc = sum(w_fc~=0,2)~=0;

% assert if corresponding bias is not zero
assert(sum(b_c1(:,nz_ch_conv1(:)==0))==0)
%assert(sum(b_c2(:,nz_ch_conv2(:)==0))==0)
assert(sum(b_c3(:,nz_ch_conv3(:)==0))==0)
assert(sum(b_c4(:,nz_ch_conv4(:)==0))==0)
assert(sum(b_fc(nz_ch_fc(:)==0))==0)

% prune!
w_c1 = w_c1(:,:,nz_ch_conv1);
b_c1 = b_c1(nz_ch_conv1);
w_c2 = w_c2(nz_ch_conv1,:,nz_ch_conv2);
b_c2 = b_c2(nz_ch_conv2);
w_c3 = w_c3(nz_ch_conv2,:,nz_ch_conv3);
b_c3 = b_c3(nz_ch_conv3);
w_c4 = w_c4(nz_ch_conv3,:,nz_ch_conv4);
b_c4 = b_c4(nz_ch_conv4);
w_fc = w_fc(nz_ch_fc,reshape(repmat(nz_ch_conv4(:),1,11).',1,[]));
b_fc = b_fc(nz_ch_fc);
%w_fc_flat = reshape(permute(w_fc,[2,1,3]),1,[],4);

%% debug
fc_len = size(w_fc,2);
fc_chin_len = size(w_fc,2)/size(w_c4,3);
assert(floor(fc_chin_len)==fc_chin_len);
%w_fc_tmp = w_fc;
for ii = 1:fc_chin_len:fc_len
w_fc(:,ii:(ii+fc_chin_len-1)) = fliplr(w_fc(:,ii:(ii+fc_chin_len-1)));
end

%% export parameters
%dut_path = '/mnt/data/loh/wvsnn/cdns_pnr/design_data/rtl/dut/';
%exportNNParams({w_c1,w_c2,w_c3,w_c4,w_fc}, {b_c1,b_c2,b_c3,b_c4,b_fc}, ntFP_param, dut_path);
%exportDWTParams(LoD, HiD, ntFP_dwt, dut_path);

%tb_path = '/mnt/data/loh/wvsnn/cdns_pnr/design_data/testbench/';
%quantPrintNReturn(in_data, ntFP_dwt, strcat(tb_path,'ecg3_in.txt'));

lmbda = quantPrintNReturn(dict.input_lmbda, numerictype(0,16,21), 'lmbda.txt'); %dict.input_lmbda;
dlta = quantPrintNReturn(dict.input_dlta, numerictype(0,16,17), 'dlta.txt'); %dict.input_dlta;

%% get ECG sequence data
% data info: f_sampl=300Hz, 16b res, dyn. range +-5mV
% label info: Normal-0, AF-1, Ohters-2, Noise-3

% database meta-data
%h5disp('physionet2017_4classes.hdf5')
info = h5info('physionet2017_4classes.hdf5');

% loop through trace data; all: 1:size(info.Groups,1)
histdat_act_c1 = [];
histdat_act_c2 = [];
histdat_act_c3 = [];
histdat_act_c4 = [];
histdat_act_fc = [];
subset_svecs_c1 = {};
subset_svecs_c2 = {};
subset_svecs_c3 = {};
subset_svecs_c4 = {};
subset_svecs_fc = {};
label_vec = zeros(4,1);
for no_sample = 1:100:size(info.Groups,1)
    %no_sample = 3;

    data = h5read('physionet2017_4classes.hdf5',strcat('/A',num2str(no_sample,'%05d'),'/data'));
    label = h5read('physionet2017_4classes.hdf5',strcat('/A',num2str(no_sample,'%05d'),'/label'));
    in_data = [data'];

    %% DWT (quantized)

    %tic;

    % quantized input
    data_quantized = quantPrintNReturn(in_data, ntFP_dwt, '');

    % filter coeff
    % quantization of intermediate values is complicated, since it requires
    % half point symmetrization (eliminate border effects for non power 2 sized frames)
    dwt_lp_coeff_quant = quantPrintNReturn(LoD, ntFP_dwt, '');
    dwt_hp_coeff_quant = quantPrintNReturn(HiD, ntFP_dwt, '');

    A1_prequant = conv(data_quantized,dwt_lp_coeff_quant,'valid');
    A1_prequant = A1_prequant(1:2:end);
    A1_quant = quantPrintNReturn(A1_prequant, ntFP_dwt, '');
    A2_prequant = conv(A1_quant,dwt_lp_coeff_quant,'valid');
    A2_prequant = A2_prequant(1:2:end);
    A2_quant = quantPrintNReturn(A2_prequant, ntFP_dwt, '');
    A3_prequant = conv(A2_quant,dwt_lp_coeff_quant,'valid');
    A3_prequant = A3_prequant(1:2:end);
    A3_quant = quantPrintNReturn(A3_prequant, ntFP_dwt, '');
    A4_prequant = conv(A3_quant,dwt_lp_coeff_quant,'valid');
    A4_prequant = A4_prequant(1:2:end);
    A4_quant = quantPrintNReturn(A4_prequant, ntFP_dwt, '');
    D4_prequant = conv(A3_quant,dwt_hp_coeff_quant,'valid');
    D4_prequant = D4_prequant(1:2:end);
    D4_quant = quantPrintNReturn(D4_prequant, ntFP_dwt, '');

    conv_in_beforenorm_quant = fi([A4_quant;D4_quant],ntFP_dwt);
    conv_in_beforenorm_quant = double(conv_in_beforenorm_quant);



    conv_in_quant = fi((conv_in_beforenorm_quant*lmbda+dlta),ntFP);
    conv_in_quant = double(conv_in_quant);

    %plot(nn_in')
    %figure
    %plot(conv_in_quant')

    %% CNN (quantized, preinitialized fifos)

    conv1_in_fifo = zeros(size(conv_in_quant,1),5);
    pool1_in_fifo = zeros(size(w_c1,3),3);
    conv2_in_fifo = zeros(size(pool1_in_fifo,1),5);
    pool2_in_fifo = zeros(size(w_c2,3),3);
    conv3_in_fifo = zeros(size(pool2_in_fifo,1),5);
    pool3_in_fifo = zeros(size(w_c3,3),3);
    conv4_in_fifo = zeros(size(pool3_in_fifo,1),5);
    pool4_in_fifo = zeros(size(w_c4,3),3);
    fc_in_fifo = zeros(size(pool4_in_fifo,1),11);
    conv1_in_act = [];
    pool1_in_act = [];
    conv2_in_act = [];
    pool2_in_act = [];
    conv3_in_act = [];
    pool3_in_act = [];
    conv4_in_act = [];
    pool4_in_act = [];
    fc_in_act = [];
    pred = [];
    conv1_svecs = [];
    conv2_svecs = [];
    conv3_svecs = [];
    conv4_svecs = [];
    fc_svecs = [];
    pool1_cnt = 0;
    pool2_cnt = 0;
    pool3_cnt = 0;
    pool4_cnt = 0;

    debug_psum = [];

    for ii = 1:(size(conv_in_quant,2)+2)

        % first two dwt values 
         if ii > (size(conv_in_quant,2)+2)
             dwt_new = [0;0];%
         elseif ii == 1
             dwt_new = [1972*2^-12;1972*2^-12];%[0;0];%
         elseif ii == 2
             dwt_new = [1975*2^-12;1978*2^-12];%[0;0];%
         else 
             dwt_new = conv_in_quant(:,ii-2);
         end

        % conv1 (push fifo)
        conv1_in_act = [conv1_in_act dwt_new];
        conv1_in_fifo = [dwt_new conv1_in_fifo(:,1:end-1)];

        % conv1 (calculate)
        %conv1_new = reshape(sum(quantPrintNReturn(sum(conv1_in_fifo .* w_c1,2),ntFP,''),1),size(pool1_in_fifo,1),[]) +b_c1';
        conv1_svec = snn_conv_val(conv1_in_fifo,w_c1,ntFP);
        conv1_svecs = cat(3,conv1_svecs,conv1_svec);
        conv1_new = conv1_svec(:,end)+b_c1';

        conv1_new = max(conv1_new,0); % ReLU
        conv1_new = min(conv1_new,quant_max_val); % Saturation
        conv1_new = quantPrintNReturn(conv1_new,ntFP,'');

        % pool1 (push fifo)
        pool1_in_act = [pool1_in_act conv1_new];
        pool1_in_fifo = [conv1_new pool1_in_fifo(:,1:end-1)];

        if pool1_cnt < 2
            pool1_cnt = pool1_cnt + 1;
            continue
        else
            pool1_cnt = 0;
            pool1_new = max(pool1_in_fifo,[],2);
            pool1_in_fifo = zeros(size(w_c1,3),3);
        end

        % conv2 (push fifo)
        conv2_in_act = [conv2_in_act pool1_new];
        conv2_in_fifo = [pool1_new conv2_in_fifo(:,1:end-1)];

        % conv2 (calculate)
        %conv2_new = reshape(sum(quantPrintNReturn(sum(conv2_in_fifo .* w_c2,2),ntFP,''),1),size(pool2_in_fifo,1),[]) +b_c2';
        conv2_svec = snn_conv_val(conv2_in_fifo,w_c2,ntFP);
        conv2_svecs = cat(3,conv2_svecs,conv2_svec);
        conv2_new = conv2_svec(:,end)+b_c2';

        conv2_new = max(conv2_new,0); % ReLU
        conv2_new = min(conv2_new,quant_max_val); % Saturation
        conv2_new = quantPrintNReturn(conv2_new,ntFP,'');

        % pool2 (push fifo)
        pool2_in_act = [pool2_in_act conv2_new];
        pool2_in_fifo = [conv2_new pool2_in_fifo(:,1:end-1)];

        if pool2_cnt < 2
            pool2_cnt = pool2_cnt + 1;
            continue
        else
            pool2_cnt = 0;
            pool2_new = max(pool2_in_fifo,[],2);
            pool2_in_fifo = zeros(size(w_c2,3),3);
        end

        % conv3 (push fifo)
        conv3_in_act = [conv3_in_act pool2_new];
        conv3_in_fifo = [pool2_new conv3_in_fifo(:,1:end-1)];

        % conv3 (calculate)
        %conv3_new = reshape(sum(quantPrintNReturn(sum(conv3_in_fifo .* w_c3,2),ntFP,''),1),size(pool3_in_fifo,1),[]) + b_c3';
        conv3_svec = snn_conv_val(conv3_in_fifo,w_c3,ntFP);
        conv3_svecs = cat(3,conv3_svecs,conv3_svec);
        conv3_new = conv3_svec(:,end)+b_c3';    

        conv3_new = max(conv3_new,0); % ReLU
        conv3_new = min(conv3_new,quant_max_val); % Saturation
        conv3_new = quantPrintNReturn(conv3_new,ntFP,'');

        % pool3 (push fifo)
        pool3_in_act = [pool3_in_act conv3_new];
        pool3_in_fifo = [conv3_new pool3_in_fifo(:,1:end-1)];

        if pool3_cnt < 2
            pool3_cnt = pool3_cnt + 1;
            continue
        else
            pool3_cnt = 0;
            pool3_new = max(pool3_in_fifo,[],2);
            pool3_in_fifo = zeros(size(w_c3,3),3);
        end

        % conv4 (push fifo)
        conv4_in_act = [conv4_in_act pool3_new];
        conv4_in_fifo = [pool3_new conv4_in_fifo(:,1:end-1)];

        % conv4 (calculate)
        %conv4_new = reshape(sum(quantPrintNReturn(sum(conv4_in_fifo .* w_c4,2),ntFP,''),1),size(pool4_in_fifo,1),[]) + b_c4';
        conv4_svec = snn_conv_val(conv4_in_fifo,w_c4,ntFP);
        conv4_svecs = cat(3,conv4_svecs,conv4_svec);
        conv4_new = conv4_svec(:,end)+b_c4';  

        conv4_new = max(conv4_new,0); % ReLU
        conv4_new = min(conv4_new,quant_max_val); % Saturation
        conv4_new = quantPrintNReturn(conv4_new,ntFP,'');

        % pool4 (push fifo)
        pool4_in_act = [pool4_in_act conv4_new];
        pool4_in_fifo = [conv4_new pool4_in_fifo(:,1:end-1)];

        if pool4_cnt < 2
            pool4_cnt = pool4_cnt + 1;
            continue
        else
            pool4_cnt = 0;
            pool4_new = max(pool4_in_fifo,[],2);
            pool4_in_fifo = zeros(size(w_c4,3),3);
        end

        % fc (push fifo)
        fc_in_act = [fc_in_act pool4_new];
        fc_in_fifo = [pool4_new fc_in_fifo(:,1:end-1)];


        % fc (calculate)
        %fc_in_fifo_aligned = fliplr(fc_in_fifo);
        %fc_svec = snn_conv_val(fc_in_fifo_aligned,w_fc,ntFP)+ b_fc';
        fc_svec = snn_conv_val(fc_in_fifo,w_fc,ntFP);
        fc_svecs = cat(3,fc_svecs,fc_svec);
        fc_new = fc_svec(:,end)+b_fc';  


        fc_new = max(fc_new,0); % ReLU
        %fc_new = min(fc_new,quant_max_val); % Saturation
        fc_new = quantPrintNReturn(fc_new,ntFP,'');

        % pred
        %debug_psum = cat(3,debug_psum,psum_vec);
        pred = [pred fc_new];
    end

    %toc;
    
    % get activations
    histdat_act_c1 = [histdat_act_c1 (2^ntFP.FractionLength+1 - bitsll(conv1_in_act,ntFP.FractionLength))];
    histdat_act_c2 = [histdat_act_c2 (2^ntFP.FractionLength+1 - bitsll(conv2_in_act,ntFP.FractionLength))];
    histdat_act_c3 = [histdat_act_c3 (2^ntFP.FractionLength+1 - bitsll(conv3_in_act,ntFP.FractionLength))];
    histdat_act_c4 = [histdat_act_c4 (2^ntFP.FractionLength+1 - bitsll(conv4_in_act,ntFP.FractionLength))];
    histdat_act_fc = [histdat_act_fc (2^ntFP.FractionLength+1 - bitsll(fc_in_act,ntFP.FractionLength))];
    
    % get svecs
    %if (mod(no_sample,1000) == 0)
%     if label ~= 0
%         subset_svecs_c1 = cat(4,subset_svecs_c1,conv1_svecs);
%         subset_svecs_c2 = cat(4,subset_svecs_c2,conv2_svecs);
%         subset_svecs_c3 = cat(4,subset_svecs_c3,conv3_svecs);
%         subset_svecs_c4 = cat(4,subset_svecs_c4,conv4_svecs);
%         subset_svecs_fc = cat(4,subset_svecs_fc,fc_svecs);
%     end
    subset_svecs_c1{(label+1)} = conv1_svecs;
    subset_svecs_c2{(label+1)} = conv2_svecs;
    subset_svecs_c3{(label+1)} = conv3_svecs;
    subset_svecs_c4{(label+1)} = conv4_svecs;
    subset_svecs_fc{(label+1)} = fc_svecs;
    label_vec((label+1)) = label_vec((label+1))+1;

end

%%
BW = 50;
BMIN = 0;
BMAX = 4097;
f=figure('Position',[1,1,600,300])
hold on 
histogram(histdat_act_c1,'BinWidth',BW,'BinLimits',[BMIN,BMAX],'normalization','probability','facealpha',.5,'edgecolor','none');
histogram(histdat_act_c2,'BinWidth',BW,'BinLimits',[BMIN,BMAX],'normalization','probability','facealpha',.5,'edgecolor','none');
histogram(histdat_act_c3,'BinWidth',BW,'BinLimits',[BMIN,BMAX],'normalization','probability','facealpha',.5,'edgecolor','none');
histogram(histdat_act_c4,'BinWidth',BW,'BinLimits',[BMIN,BMAX],'normalization','probability','facealpha',.5,'edgecolor','none');
histogram(histdat_act_fc,'BinWidth',BW,'BinLimits',[BMIN,BMAX],'normalization','probability','facealpha',.5,'edgecolor','none');



xlabel('Timestep t')
ylabel('Probability p')
%ylim([5*10^-1,10^5])
xlim([0,4096])
grid on; grid minor;
box on;
legend('L1','L2','L3','L4','FC','Location','northwest')
set(gca,'YScale','log')

%printpdf(f,'act_stats.pdf')

%%
BW = 100;
BMIN = 0;
BMAX = 4097;
f=figure('Position',[1,1,600,300])
[n1,e1] = histcounts(histdat_act_c1,'BinWidth',BW,'BinLimits',[BMIN,BMAX],'normalization','probability');
[n2,e2] = histcounts(histdat_act_c2,'BinWidth',BW,'BinLimits',[BMIN,BMAX],'normalization','probability');
[n3,e3] = histcounts(histdat_act_c3,'BinWidth',BW,'BinLimits',[BMIN,BMAX],'normalization','probability');
[n4,e4] = histcounts(histdat_act_c4,'BinWidth',BW,'BinLimits',[BMIN,BMAX],'normalization','probability');
[n5,e5] = histcounts(histdat_act_fc,'BinWidth',BW,'BinLimits',[BMIN,BMAX],'normalization','probability');

edges = e1(2:end) - (e1(2)-e1(1))/2;
bar(edges,[n1;n2;n3;n4;n5]/5','stacked')

xlabel('Timestep')
ylabel('Probability p')
%ylim([5*10^-1,10^5])
xlim([0,4096])
grid on; grid minor;
box on;
legend('L1','L2','L3','L4','FC','Location','northwest')
set(gca,'YScale','log')
set(gca, 'FontName', 'Arial')
printpdf(f,'act_stats.pdf')

%% plot activation statistics
BW = 50;
BMIN = 0;
BMAX = 4096;

figure('Position',[1,1,1500,750]);
sgtitle("SNN activation statistics; Subset of CinC'17 (~80+ samples); time resolution 0-4095 (integer); BIN=50 timesteps")

subplot(2,5,1);
histogram(histdat_act_c1,'BinWidth',BW,'BinLimits',[BMIN,BMAX]);
title('C1 input activations')
xlabel('timestep t')
ylabel('occurences M')
ylim([5*10^-1,10^5])
grid on; grid minor;
set(gca,'YScale','log')
subplot(2,5,2);
histogram(histdat_act_c2,'BinWidth',BW,'BinLimits',[BMIN,BMAX]);
title('C2 input activations')
xlabel('timestep t')
ylabel('occurences M')
ylim([5*10^-1,10^5])
grid on; grid minor;
set(gca,'YScale','log')
subplot(2,5,3);
histogram(histdat_act_c3,'BinWidth',BW,'BinLimits',[BMIN,BMAX]);
title('C3 input activations')
xlabel('timestep t')
ylabel('occurences M')
ylim([5*10^-1,10^5])
grid on; grid minor;
set(gca,'YScale','log')
subplot(2,5,4);
histogram(histdat_act_c4,'BinWidth',BW,'BinLimits',[BMIN,BMAX]);
title('C4 input activations')
xlabel('timestep t')
ylabel('occurences M')
ylim([5*10^-1,10^5])
grid on; grid minor;
set(gca,'YScale','log')
subplot(2,5,5);
histogram(histdat_act_fc,'BinWidth',BW,'BinLimits',[BMIN,BMAX]);
title('FC input activations')
xlabel('timestep t')
ylabel('occurences M')
ylim([5*10^-1,10^5])
grid on; grid minor;
set(gca,'YScale','log')
subplot(2,5,6:10);



%     %
%     %figure
%     q_steps = 0.01;
%     alpharange = 0.95;
%     color = [0 0 1];
%     hold on
%     
%     curve1 = zeros(5,1);
%     curve2 = zeros(5,1);
%     curve1_prev = [];
%     curve2_prev = [];
%     for ii=1:-q_steps:0
%         p = ii/2;
%         alpha = ii * alpharange + (1-alpharange);
%         dat = quantile(histdat_act_c1(:),[p 1-p]);
%         curve1(1,1) = dat(1);
%         curve2(1,1) = dat(2);
%         dat = quantile(histdat_act_c2(:),[p 1-p]);
%         curve1(2,1) = dat(1);
%         curve2(2,1) = dat(2);
%         dat = quantile(histdat_act_c3(:),[p 1-p]);
%         curve1(3,1) = dat(1);
%         curve2(3,1) = dat(2);
%         dat = quantile(histdat_act_c4(:),[p 1-p]);
%         curve1(4,1) = dat(1);
%         curve2(4,1) = dat(2);
%         dat = quantile(histdat_act_fc(:),[p 1-p]);
%         curve1(5,1) = dat(1);
%         curve2(5,1) = dat(2);
% 
%         if ii~=1
%             % Find area coordinates.
%             inBetweenRegionX = [1:length(curve1), length(curve1_prev):-1:1]';
%             inBetweenRegionY = [curve1; flipud(curve1_prev)];
%             % Display the area first so it will be in the background.
%             fill(gca,inBetweenRegionX, inBetweenRegionY,color,'facealpha',alpha,'LineStyle','none');
%             % Find area coordinates.
%             inBetweenRegionX = [1:length(curve2), length(curve2_prev):-1:1]';
%             inBetweenRegionY = [curve2; flipud(curve2_prev)];
%             % Display the area first so it will be in the background.
%             fill(inBetweenRegionX, inBetweenRegionY,color,'facealpha',alpha,'LineStyle','none');            
%         
%         else
%             plot(gca,[median(histdat_act_c1(:)),median(histdat_act_c2(:)),median(histdat_act_c3(:)),median(histdat_act_c4(:)),median(histdat_act_fc(:))],'Color',color);
%         end
%             
%         curve1_prev = curve1;
%         curve2_prev = curve2;
%     end
    
boxplot([histdat_act_c1(:);histdat_act_c2(:);histdat_act_c3(:);histdat_act_c4(:);histdat_act_fc(:)],[repmat({'C1'},prod(size(histdat_act_c1)),1);repmat({'C2'},prod(size(histdat_act_c2)),1);repmat({'C3'},prod(size(histdat_act_c3)),1);repmat({'C4'},prod(size(histdat_act_c4)),1);repmat({'FC'},prod(size(histdat_act_fc)),1)])
% hold off
title('Activation distribution per layer')
xlabel('SNN Layer')
ylabel('timestep t')
%ylim([5*10^-1,10^5])
grid on; grid minor;

%% sample traces of svecs include one of each class (1:N, 2:AF, 3:O, 4:~)
sample_trace_idx = 1;

%%
figure('Position',[1,1,1500,750])
sgtitle("SNN acc reg statistics; Single sample of CinC'17; time resolution 0-4095 (integer); before saturation")

subplot(2,5,1);
plot_distribution_over_time(reshape(permute(subset_svecs_c1{sample_trace_idx}(1,:,:),[1,3,2]),[],size(subset_svecs_c1{sample_trace_idx},2)))
title('C1 acc reg (neuron no 1)')
xlabel('timestep t')
ylabel('value s')
%ylim([-1,1])
subplot(2,5,2);
plot_distribution_over_time(reshape(permute(subset_svecs_c2{sample_trace_idx}(1,:,:),[1,3,2]),[],size(subset_svecs_c2{sample_trace_idx},2)))
title('C2 acc reg (neuron no 1)')
xlabel('timestep t')
ylabel('value s')
%ylim([-1,1])
subplot(2,5,3);
plot_distribution_over_time(reshape(permute(subset_svecs_c3{sample_trace_idx}(1,:,:),[1,3,2]),[],size(subset_svecs_c3{sample_trace_idx},2)))
title('C3 acc reg (neuron no 1)')
xlabel('timestep t')
ylabel('value s')
%ylim([-1,1])
subplot(2,5,4);
plot_distribution_over_time(reshape(permute(subset_svecs_c4{sample_trace_idx}(1,:,:),[1,3,2]),[],size(subset_svecs_c4{sample_trace_idx},2)))
title('C4 acc reg (neuron no 1)')
xlabel('timestep t')
ylabel('value s')
%ylim([-1,1])
subplot(2,5,5);
plot_distribution_over_time(reshape(permute(subset_svecs_fc{sample_trace_idx}(1,:,:),[1,3,2]),[],size(subset_svecs_fc{sample_trace_idx},2)))
title('FC acc reg (all neurons)')
xlabel('timestep t')
ylabel('value s')
%ylim([-1,1])
subplot(2,5,6);
plot_distribution_over_time(reshape(permute(subset_svecs_c1{sample_trace_idx},[1,3,2]),[],size(subset_svecs_c1{sample_trace_idx},2)))
title('C1 acc reg (all neurons)')
xlabel('timestep t')
ylabel('value s')
%ylim([-1,1])
subplot(2,5,7);
plot_distribution_over_time(reshape(permute(subset_svecs_c2{sample_trace_idx},[1,3,2]),[],size(subset_svecs_c2{sample_trace_idx},2)))
title('C2 acc reg (all neurons)')
xlabel('timestep t')
ylabel('value s')
%ylim([-1,1])
subplot(2,5,8);
plot_distribution_over_time(reshape(permute(subset_svecs_c3{sample_trace_idx},[1,3,2]),[],size(subset_svecs_c3{sample_trace_idx},2)))
title('C3 acc reg (all neurons)')
xlabel('timestep t')
ylabel('value s')
%ylim([-1,1])
subplot(2,5,9);
plot_distribution_over_time(reshape(permute(subset_svecs_c4{sample_trace_idx},[1,3,2]),[],size(subset_svecs_c4{sample_trace_idx},2)))
title('C4 acc reg (all neurons)')
xlabel('timestep t')
ylabel('value s')
%ylim([-1,1])
subplot(2,5,10);
plot_distribution_over_time(reshape(permute(subset_svecs_fc{sample_trace_idx},[1,3,2]),[],size(subset_svecs_fc{sample_trace_idx},2)))
title('FC acc reg (all neurons)')
xlabel('timestep t')
ylabel('value s')
%ylim([-1,1])

%%

% % convert acc reg value into quantized data over time (for each conv calculation of ONE ECG sequence)
% flat_vec = reshape(permute(subset_svecs_c4{sample_trace_idx},[1,3,2]),[],size(subset_svecs_c4{sample_trace_idx},2));
% flat_vec_bin = fi(bitsll(flat_vec,ntFP.FractionLength*2),numerictype(1,24,0));
% flat_mat = bin(flat_vec_bin(:))-'0';
% kk = reshape(repmat(1:2^ntFP.FractionLength,size(flat_vec,1),1),[],1);
% 
% % convert quantized data into occurences of 1's per time step
% heatmap_dat = cell2mat(accumarray(kk,1:numel(kk),[],@(x) {sum(flat_mat(x,:),1)}))';
% 
% % convert quantized data into occurences of bit transitions (0->1 & 1->0)
% % per time step
% flat_mat_diff = abs(cell2mat(accumarray(kk,1:numel(kk),[],@(x) {diff(flat_mat(x,:),1)})));
% kk_diff = reshape(repmat(1:2^ntFP.FractionLength,(size(flat_vec,1)-1),1),[],1);
% heatmap_diff_dat = cell2mat(accumarray(kk_diff,1:numel(kk_diff),[],@(x) {sum(flat_mat_diff(x,:),1)}))';
[heatmap_c1_dat, heatmap_c1_diff_dat] = calc_acc_reg_statistics(subset_svecs_c1{sample_trace_idx},ntFP);
[heatmap_c2_dat, heatmap_c2_diff_dat] = calc_acc_reg_statistics(subset_svecs_c2{sample_trace_idx},ntFP);
[heatmap_c3_dat, heatmap_c3_diff_dat] = calc_acc_reg_statistics(subset_svecs_c3{sample_trace_idx},ntFP);
[heatmap_c4_dat, heatmap_c4_diff_dat] = calc_acc_reg_statistics(subset_svecs_c4{sample_trace_idx},ntFP);
[heatmap_fc_dat, heatmap_fc_diff_dat] = calc_acc_reg_statistics(subset_svecs_fc{sample_trace_idx},ntFP);

XLabels = 1:2^ntFP.FractionLength;
CustomXLabels = string(XLabels);
CustomXLabels(mod(XLabels,100) ~= 0) = " ";

%%
figure('Position',[1,1,1500,750])
sgtitle('C1 Acc Reg - All neurons cumulative statistics')
subplot(2,1,1)
h=heatmap(heatmap_c1_dat,'GridVisible','off','CellLabelColor','none','ColorMap',hot,'XDisplayLabels',CustomXLabels,'YDisplayLabels',0:2*ntFP.FractionLength-1);
title('Occurences of 1s')
caxis([0, prod(size(subset_svecs_c1{sample_trace_idx},[1,3]))])
h.NodeChildren(3).YDir='normal';
xlabel('timestep t')
ylabel('bit significance')
subplot(2,1,2)
h=heatmap(heatmap_c1_diff_dat,'GridVisible','off','CellLabelColor','none','ColorMap',hot,'XDisplayLabels',CustomXLabels,'YDisplayLabels',0:2*ntFP.FractionLength-1);
title('Occurences of transitions (0->1 & 1->0)')
caxis([0, prod(size(subset_svecs_c1{sample_trace_idx},[1,3]))])
h.NodeChildren(3).YDir='normal';
xlabel('timestep t')
ylabel('bit significance')
% figure('Position',[1,1,1500,750])
% sgtitle('C2 Acc Reg - All neurons cumulative statistics')
% subplot(2,1,1)
% h=heatmap(heatmap_c2_dat,'GridVisible','off','CellLabelColor','none','ColorMap',hot,'XDisplayLabels',CustomXLabels,'YDisplayLabels',0:2*ntFP.FractionLength-1);
% title('Occurences of 1s')
% caxis([0, prod(size(subset_svecs_c2{sample_trace_idx},[1,3]))])
% h.NodeChildren(3).YDir='normal';
% xlabel('timestep t')
% ylabel('bit significance')
% subplot(2,1,2)
% h=heatmap(heatmap_c2_diff_dat,'GridVisible','off','CellLabelColor','none','ColorMap',hot,'XDisplayLabels',CustomXLabels,'YDisplayLabels',0:2*ntFP.FractionLength-1);
% title('Occurences of transitions (0->1 & 1->0)')
% caxis([0, prod(size(subset_svecs_c2{sample_trace_idx},[1,3]))])
% h.NodeChildren(3).YDir='normal';
% xlabel('timestep t')
% ylabel('bit significance')
% figure('Position',[1,1,1500,750])
% sgtitle('C3 Acc Reg - All neurons cumulative statistics')
% subplot(2,1,1)
% h=heatmap(heatmap_c3_dat,'GridVisible','off','CellLabelColor','none','ColorMap',hot,'XDisplayLabels',CustomXLabels,'YDisplayLabels',0:2*ntFP.FractionLength-1);
% title('Occurences of 1s')
% caxis([0, prod(size(subset_svecs_c3{sample_trace_idx},[1,3]))])
% h.NodeChildren(3).YDir='normal';
% xlabel('timestep t')
% ylabel('bit significance')
% subplot(2,1,2)
% h=heatmap(heatmap_c3_diff_dat,'GridVisible','off','CellLabelColor','none','ColorMap',hot,'XDisplayLabels',CustomXLabels,'YDisplayLabels',0:2*ntFP.FractionLength-1);
% title('Occurences of transitions (0->1 & 1->0)')
% caxis([0, prod(size(subset_svecs_c3{sample_trace_idx},[1,3]))])
% h.NodeChildren(3).YDir='normal';
% xlabel('timestep t')
% ylabel('bit significance')
% figure('Position',[1,1,1500,750])
% sgtitle('C4 Acc Reg - All neurons cumulative statistics')
% subplot(2,1,1)
% h=heatmap(heatmap_c4_dat,'GridVisible','off','CellLabelColor','none','ColorMap',hot,'XDisplayLabels',CustomXLabels,'YDisplayLabels',0:2*ntFP.FractionLength-1);
% title('Occurences of 1s')
% caxis([0, prod(size(subset_svecs_c4{sample_trace_idx},[1,3]))])
% h.NodeChildren(3).YDir='normal';
% xlabel('timestep t')
% ylabel('bit significance')
% subplot(2,1,2)
% h=heatmap(heatmap_c4_diff_dat,'GridVisible','off','CellLabelColor','none','ColorMap',hot,'XDisplayLabels',CustomXLabels,'YDisplayLabels',0:2*ntFP.FractionLength-1);
% title('Occurences of transitions (0->1 & 1->0)')
% caxis([0, prod(size(subset_svecs_c4{sample_trace_idx},[1,3]))])
% h.NodeChildren(3).YDir='normal';
% xlabel('timestep t')
% ylabel('bit significance')
% figure('Position',[1,1,1500,750])
% sgtitle('FC Acc Reg - All neurons cumulative statistics')
% subplot(2,1,1)
% h=heatmap(heatmap_fc_dat,'GridVisible','off','CellLabelColor','none','ColorMap',hot,'XDisplayLabels',CustomXLabels,'YDisplayLabels',0:2*ntFP.FractionLength-1);
% title('Occurences of 1s')
% caxis([0, prod(size(subset_svecs_fc{sample_trace_idx},[1,3]))])
% h.NodeChildren(3).YDir='normal';
% xlabel('timestep t')
% ylabel('bit significance')
% subplot(2,1,2)
% h=heatmap(heatmap_fc_diff_dat,'GridVisible','off','CellLabelColor','none','ColorMap',hot,'XDisplayLabels',CustomXLabels,'YDisplayLabels',0:2*ntFP.FractionLength-1);
% title('Occurences of transitions (0->1 & 1->0)')
% caxis([0, prod(size(subset_svecs_fc{sample_trace_idx},[1,3]))])
% h.NodeChildren(3).YDir='normal';
% xlabel('timestep t')
% ylabel('bit significance')
%%

figure('Position',[1,1,1500,750])
sgtitle("SNN acc reg statistics (per bit); Single sample of CinC'17; reg width 2*f=24; before saturation")
subplot(2,5,1)
bar(0:2*ntFP.FractionLength-1,sum(heatmap_c1_dat,2))
title('C1 - 1s')
xlabel('bit significance')
ylabel('occurences')
subplot(2,5,2)
bar(0:2*ntFP.FractionLength-1,sum(heatmap_c2_dat,2))
title('C2 - 1s')
xlabel('bit significance')
ylabel('occurences')
subplot(2,5,3)
bar(0:2*ntFP.FractionLength-1,sum(heatmap_c3_dat,2))
title('C3 - 1s')
xlabel('bit significance')
ylabel('occurences')
subplot(2,5,4)
bar(0:2*ntFP.FractionLength-1,sum(heatmap_c4_dat,2))
title('C4 - 1s')
xlabel('bit significance')
ylabel('occurences')
subplot(2,5,5)
bar(0:2*ntFP.FractionLength-1,sum(heatmap_fc_dat,2))
title('FC - 1s')
xlabel('bit significance')
ylabel('occurences')
subplot(2,5,6)
bar(0:2*ntFP.FractionLength-1,sum(heatmap_c1_diff_dat,2))
title('C1 - Transitions')
xlabel('bit significance')
ylabel('occurences')
subplot(2,5,7)
bar(0:2*ntFP.FractionLength-1,sum(heatmap_c2_diff_dat,2))
title('C2 - Transitions')
xlabel('bit significance')
ylabel('occurences')
subplot(2,5,8)
bar(0:2*ntFP.FractionLength-1,sum(heatmap_c3_diff_dat,2))
title('C3 - Transitions')
xlabel('bit significance')
ylabel('occurences')
subplot(2,5,9)
bar(0:2*ntFP.FractionLength-1,sum(heatmap_c4_diff_dat,2))
title('C4 - Transitions')
xlabel('bit significance')
ylabel('occurences')
subplot(2,5,10)
bar(0:2*ntFP.FractionLength-1,sum(heatmap_fc_diff_dat,2))
title('FC - Transitions')
xlabel('bit significance')
ylabel('occurences')

%%
[heatmap_abs_c1_dat, heatmap_abs_c1_diff_dat] = calc_acc_reg_statistics(abs(subset_svecs_c1{sample_trace_idx}),ntFP);
[heatmap_abs_c2_dat, heatmap_abs_c2_diff_dat] = calc_acc_reg_statistics(abs(subset_svecs_c2{sample_trace_idx}),ntFP);
[heatmap_abs_c3_dat, heatmap_abs_c3_diff_dat] = calc_acc_reg_statistics(abs(subset_svecs_c3{sample_trace_idx}),ntFP);
[heatmap_abs_c4_dat, heatmap_abs_c4_diff_dat] = calc_acc_reg_statistics(abs(subset_svecs_c4{sample_trace_idx}),ntFP);
[heatmap_abs_fc_dat, heatmap_abs_fc_diff_dat] = calc_acc_reg_statistics(abs(subset_svecs_fc{sample_trace_idx}),ntFP);

%%
figure('Position',[1,1,1500,750])
sgtitle('C1 Acc Reg - All neurons cumulative statistics (after L1 norm)')
subplot(2,1,1)
h=heatmap(heatmap_abs_c1_dat,'GridVisible','off','CellLabelColor','none','ColorMap',hot,'XDisplayLabels',CustomXLabels,'YDisplayLabels',0:2*ntFP.FractionLength-1);
title('Occurences of 1s')
caxis([0, prod(size(subset_svecs_c1{sample_trace_idx},[1,3]))])
h.NodeChildren(3).YDir='normal';
xlabel('timestep t')
ylabel('bit significance')
subplot(2,1,2)
h=heatmap(heatmap_abs_c1_diff_dat,'GridVisible','off','CellLabelColor','none','ColorMap',hot,'XDisplayLabels',CustomXLabels,'YDisplayLabels',0:2*ntFP.FractionLength-1);
title('Occurences of transitions (0->1 & 1->0)')
caxis([0, prod(size(subset_svecs_c1{sample_trace_idx},[1,3]))])
h.NodeChildren(3).YDir='normal';
xlabel('timestep t')
ylabel('bit significance')

%%

figure('Position',[1,1,1500,750])
sgtitle("SNN acc reg statistics (hypothetical - abs value only); Single sample of CinC'17; reg width 2*f=24; before saturation")
subplot(2,5,1)
bar(0:2*ntFP.FractionLength-1,sum(heatmap_abs_c1_dat,2))
title('C1 - 1s')
xlabel('bit significance')
ylabel('occurences')
subplot(2,5,2)
bar(0:2*ntFP.FractionLength-1,sum(heatmap_abs_c2_dat,2))
title('C2 - 1s')
xlabel('bit significance')
ylabel('occurences')
subplot(2,5,3)
bar(0:2*ntFP.FractionLength-1,sum(heatmap_abs_c3_dat,2))
title('C3 - 1s')
xlabel('bit significance')
ylabel('occurences')
subplot(2,5,4)
bar(0:2*ntFP.FractionLength-1,sum(heatmap_abs_c4_dat,2))
title('C4 - 1s')
xlabel('bit significance')
ylabel('occurences')
subplot(2,5,5)
bar(0:2*ntFP.FractionLength-1,sum(heatmap_abs_fc_dat,2))
title('FC - 1s')
xlabel('bit significance')
ylabel('occurences')
subplot(2,5,6)
bar(0:2*ntFP.FractionLength-1,sum(heatmap_abs_c1_diff_dat,2))
title('C1 - Transitions')
xlabel('bit significance')
ylabel('occurences')
subplot(2,5,7)
bar(0:2*ntFP.FractionLength-1,sum(heatmap_abs_c2_diff_dat,2))
title('C2 - Transitions')
xlabel('bit significance')
ylabel('occurences')
subplot(2,5,8)
bar(0:2*ntFP.FractionLength-1,sum(heatmap_abs_c3_diff_dat,2))
title('C3 - Transitions')
xlabel('bit significance')
ylabel('occurences')
subplot(2,5,9)
bar(0:2*ntFP.FractionLength-1,sum(heatmap_abs_c4_diff_dat,2))
title('C4 - Transitions')
xlabel('bit significance')
ylabel('occurences')
subplot(2,5,10)
bar(0:2*ntFP.FractionLength-1,sum(heatmap_abs_fc_diff_dat,2))
title('FC - Transitions')
xlabel('bit significance')
ylabel('occurences')
