%% Author: Johnson Loh; Date: 01.06.2021
clear all; close all; clc; fclose('all');

%% No Adds
epsilon = 100;

n=12;
N=5;
C=1:256;
out_y = [];
out_x = [];
out_n = [];
err = [];
err_all = [];

for n=2:32
    
    no_add_cnn = n * N * C + 1;
    no_add_snn = 2^n + N * C + 1;
    diff = abs(no_add_cnn - no_add_snn);
    [d_val,idx] = min(diff);

    if d_val < epsilon
        out_y = [out_y no_add_snn(idx)];
        out_n = [out_n n];
        out_x = [out_x C(idx)];
        err = [err d_val];
    end
    err_all = [err_all d_val];
end

plot(out_x,out_y)

text(out_x,out_y,num2str(out_n'))

%% plot intersect
close all;
n = 0:1:32;
N=5;
C=32;

f=figure('position',[0 0 400 300]);
hold on
ylim([0,20000]);
xlim([1,16]);
grid on
%grid minor
ax=gca
%ax.XAxis.MinorTickValues = 1:1:16;
ax.XAxis.TickValues = 1:1:16;
ax.YAxis.TickValues = 0:2500:20000;
xlabel('Number of bits')
ylabel('Number of additions')
%title('No. ADD Ops for a 1D conv. of kernel size 5');

%chan_vec = [32 64 128 256];
chan_vec = [32 64 128 256];
color_vec = [1 0 0; 0 1 0; 0 0 1; 0 0 0];

for ii=1:size(chan_vec,2)
C=chan_vec(ii);
X = N*C;
no_add_cnn = n * X + 1;
no_add_snn = 2.^n + X + 1;

[intersectx,intersecty] = polyxpoly(n,no_add_cnn,n,no_add_snn);
validx = n(n >= min(intersectx) & n < max(intersectx));
validy = no_add_snn(n >= min(intersectx) & n < max(intersectx));

%figure
plot(n,no_add_cnn,'--','color',color_vec(ii,:));
%hold on
plot(n,no_add_snn,'-.','color',color_vec(ii,:));
%ylim([0,max(intersecty)*1.1]);
scatter(validx,validy,50,color_vec(ii,:),'x','LineWidth',1.5);
end

h = zeros(3, 1);
%h(1) = plot(NaN,NaN,'-k');
%h(2) = plot(NaN,NaN,'--','color',[1 1 1]*0.5);
%h(3) = plot(NaN,NaN,'-.','color',[1 1 1]*0.5);
%legend(h, 'C = 256','MAC-based','Time-encoded', 'Location','northwest');
h(1) = plot(NaN,NaN,'-r');
h(2) = plot(NaN,NaN,'-g');
h(3) = plot(NaN,NaN,'-b');
h(4) = plot(NaN,NaN,'-k');
h(5) = plot(NaN,NaN,'--','color',[1 1 1]*0.5);
h(6) = plot(NaN,NaN,'-.','color',[1 1 1]*0.5);
legend(h, 'C = 32', 'C = 64','C = 128','C = 256','MAC-based','Time-encoded', 'Location','northwest');


box on

printpdf(f,'num_add_timecode.pdf')