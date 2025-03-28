%% Visualization of the state of memory needed to process one set of independent variables in a ANN layer
% $Author: Johnson Loh $  $Date: 2019/12/18 $ $Revision: 0.1 $
% Copyright: 
clear; clc; close all;

%% Small GA run (ref)
tmp.name = 'testlayer';
tmp.type = 'Conv2d';
tmp.dat = [2 2 3 3 1 1 0 0];
tmp.dim_in = [7 7];
tmp.ch_in = 2; tmp.dim_out = [5 5]; tmp.ch_out = 2;
tmp.num_val_in = prod([(tmp.dim_in + 2*tmp.dat(7:8)) tmp.ch_in]);%prod(tmp.dim_in) * tmp.ch_in;
tmp.num_val_out = prod(tmp.dim_out) * tmp.ch_out;
tmp.num_val_param_per_out = prod(tmp.dat(3:4)) * tmp.ch_in;
tmp.mem_paraminternal = prod(tmp.dat(3:4)) * tmp.ch_in * tmp.ch_out;

elem = 3*3*2*2;
no_pe = 1;
tic
cost_smalllayer_ref_weightstat = optim_get_seq_cost(no_pe, {1:elem}, 2, tmp);
toc
% tic
% cost_smalllayer_ref_weightstat_par = optim_get_seq_cost_par(pe, {1:elem}, 2, tmp);
% toc
optimal_sequence = [1 2 3 6 5 4 7 8 9 10 11 12 15 14 13 16 17 18 36 35 34 31 32 33 30 29 28 27 26 25 22 23 24 21 20 19];
tic
cost_smalllayer_ref_weightstat_opt = optim_get_seq_cost(no_pe, {optimal_sequence}, 2, tmp);
toc

%plot_seq_weight({optimal_sequence},tmp)

%% Small GA run (opt)
% %delete(gcp)
% FitnessFcn = @(x)optim_get_seq_cost(pe,x,2,tmp);
% options = optimoptions(@ga, 'PopulationType', 'custom','InitialPopulationRange', ...
%                             [1;elem]);
% options = optimoptions(options,'CreationFcn',@optim_create_permutations, ...
%                         'CrossoverFcn',@optim_crossover_permutation, ...
%                         'MutationFcn',@optim_mutate_permutation, ...
%                         'MaxGenerations',2520,'PopulationSize',1000, ...
%                         'MaxStallGenerations',100,'UseVectorized', true,'UseParallel', false);
% 
% numberOfVariables = elem;
% 
% % Profile Optimization
% disp('Start Vectorized GA Optimization with pop_size = 1000')
% tic
% [seq_smalllayer,cost_smalllayer,reason,opt_output_smalllayer] = ...
%     ga(FitnessFcn,numberOfVariables,[],[],[],[],[],[],[],options);
% toc
% disp('End Vectorized GA Optimization with pop_size = 1000')
% %plot_seq(seq_smalllayer,tmp)

%% Small GA run (opt)
% %delete(gcp)
% %parpool('local');
% 
% max_pe_size = [7*7*2 3*3*2*2 5*5*2];
% cost_pesweep = cell(max(max_pe_size),3);
% 
% for dataflow = 1:3
% %for dataflow = 2
%     disp(['Dataflow No.: ' num2str(dataflow)])
%     for pe = 1:max_pe_size(dataflow)
%         %FitnessFcn = @(x)optim_get_seq_cost_par(pe,x,dataflow,tmp);
%         FitnessFcn = @(x)optim_get_seq_cost(pe,x,dataflow,tmp);
%         options = optimoptions(@ga, 'PopulationType', 'custom','InitialPopulationRange', ...
%                                     [1;max_pe_size(dataflow)]);
%         options = optimoptions(options,'CreationFcn',@optim_create_permutations, ...
%                                 'CrossoverFcn',@optim_crossover_permutation, ...
%                                 'MutationFcn',@optim_mutate_permutation, ...
%                                 'MaxGenerations',2500,'PopulationSize',2000, ...
%                                 'MaxStallGenerations',100,'UseParallel',true, ...
%                                 'UseVectorized', false);
% 
%         numberOfVariables = max_pe_size(dataflow);
% 
%         % Profile Optimization
%         disp(['Start Parallelized GA Optimization with pop_size = 2000 and no_pe = ' num2str(pe)])
%         tic
%         [seq_smalllayer,cost_smalllayer,reason,opt_output_smalllayer] = ...
%             ga(FitnessFcn,numberOfVariables,[],[],[],[],[],[],[],options);
%         toc
% 
%         cost_pesweep{pe,dataflow} = [seq_smalllayer,cost_smalllayer,reason,opt_output_smalllayer];
%         %disp(['End Parallelized GA Optimization with pop_size = 1000 and no_pe = ' num2str(pe)])
%     end
% end
% delete(gcp)
% save('workspace_chkpt')
%plot_seq_weight(seq_smalllayer,tmp)
%%

f=figure('Position',[1,1,500,300])
result_weightstat = [cost_pesweep{:,2}];
cost_weightstat = [result_weightstat{2:4:end}];
plot(1:10,cost_weightstat(1:10)/cost_weightstat(1),'bx-','linewidth',2,'markersize',10);
xlim([1,10])
grid on
%grid minor

%title({'Case Study 1: Memory can only hold data from previous iteration and new data' 'Cost = Count of Values loaded' '(Optimized Sequence via Genetic Algorithm)'})
ylabel('Relative Cost')
xlabel('No. PEs')

printpdf(f,'3_ga_opt_pesweep.pdf')

%% Analysis
result_instat = [cost_pesweep{:,1}];
result_weightstat = [cost_pesweep{:,2}];
result_outstat = [cost_pesweep{:,3}];

cost_instat = [result_instat{2:4:end}];
cost_weightstat = [result_weightstat{2:4:end}];
cost_outstat = [result_outstat{2:4:end}];

figure
hold on
plot(cost_instat,'rx-')
plot(cost_weightstat,'g+-')
plot(cost_outstat,'bo-')
grid on
grid minor
title({'Case Study 1: Memory can only hold data from previous iteration and new data' 'Cost = Count of Values loaded' '(Optimized Sequence via Genetic Algorithm)'})
legend('Input stationary','Weight stationary','Output stationary')
ylabel('Cost (No. Values)')
xlabel('No. PEs')

%% Plot one run

dataflow = 2;
pe = 1;
max_pe_size = [7*7*2 3*3*2*2 5*5*2];
cost_pesweep = cell(max(max_pe_size),3);

%FitnessFcn = @(x)optim_get_seq_cost_par(pe,x,dataflow,tmp);
FitnessFcn = @(x)optim_get_seq_cost(pe,x,dataflow,tmp);
options = optimoptions(@ga, 'PopulationType', 'custom','InitialPopulationRange', ...
                            [1;max_pe_size(dataflow)]);
options = optimoptions(options,'PlotFcn',{@gaplotbestf2,@gaplotscorediversity}, ... %{@gaplotbestf,@gaplotscorediversity}
                        'CreationFcn',@optim_create_permutations, ...
                        'CrossoverFcn',@optim_crossover_permutation, ...
                        'MutationFcn',@optim_mutate_permutation, ...
                        'MaxGenerations',500,'PopulationSize',2000, ...
                        'MaxStallGenerations',50,'UseParallel',true, ...
                        'UseVectorized', false);

numberOfVariables = max_pe_size(dataflow);

% Profile Optimization
disp(['Start Parallelized GA Optimization with pop_size = 2000 and no_pe = ' num2str(pe)])
tic
[seq_smalllayer,cost_smalllayer,reason,opt_output_smalllayer] = ...
    ga(FitnessFcn,numberOfVariables,[],[],[],[],[],[],[],options);
toc

%%

xlim(gca,[0,100])
hold on
plot(0:1:100,ones(1,101)*cost_smalllayer_ref_weightstat,'r-','LineWidth',2)
legend('Best fitness','Mean fitness','Worst fitness','Reference Sequence')

f=gcf;
printpdf(f,'3_ga_opt_fitness.pdf')