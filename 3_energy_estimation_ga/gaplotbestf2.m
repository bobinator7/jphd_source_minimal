function state = gaplotbestf2(options,state,flag)
%GAPLOTBESTF Plots the best score and the mean score.
%   STATE = GAPLOTBESTF(OPTIONS,STATE,FLAG) plots the best score as well
%   as the mean of the scores.
%
%   Example:
%    Create an options structure that will use GAPLOTBESTF
%    as the plot function
%     options = optimoptions('ga','PlotFcn',@gaplotbestf);

%   Copyright 2003-2016 The MathWorks, Inc.

if size(state.Score,2) > 1
    msg = getString(message('globaloptim:gaplotcommon:PlotFcnUnavailable','gaplotbestf2'));
    title(msg,'interp','none');
    return;
end

switch flag
    case 'init'
        hold on;
        set(gca,'xlim',[0,options.MaxGenerations]);
        xlabel('Generation','interp','none');
        ylabel('Fitness value','interp','none');
        plotBest = plot(state.Generation,min(state.Score),'.k');
        set(plotBest,'Tag','gaplotbestf');
        plotMean = plot(state.Generation,meanf(state.Score),'.b');
        set(plotMean,'Tag','gaplotmean');
        plotWorst = plot(state.Generation,max(state.Score),'.k');
        set(plotWorst,'Tag','gaplotworstf');
        title(['Best: ',' Mean: ',' Worst: '],'interp','none')
    case 'iter'
        best  = min(state.Score);
        m     = meanf(state.Score);
        worst = max(state.Score);
        plotBest = findobj(get(gca,'Children'),'Tag','gaplotbestf');
        plotMean = findobj(get(gca,'Children'),'Tag','gaplotmean');
        plotWorst = findobj(get(gca,'Children'),'Tag','gaplotworstf');
        newX = [get(plotBest,'Xdata') state.Generation];
        newY = [get(plotBest,'Ydata') best];
        set(plotBest,'Xdata',newX, 'Ydata',newY);
        newY = [get(plotMean,'Ydata') m];
        set(plotMean,'Xdata',newX, 'Ydata',newY);
        newY = [get(plotWorst,'Ydata') worst];
        set(plotWorst,'Xdata',newX, 'Ydata',newY);
        set(get(gca,'Title'),'String',sprintf('Best: %g Mean: %g Worst: %g',best,m,worst));
    case 'done'
        LegnD = legend('Best/Worst fitness','Mean fitness');
        set(LegnD,'FontSize',8);
        hold off;
end

%------------------------------------------------
function m = meanf(x)
nans = isnan(x);
x(nans) = 0;
n = sum(~nans);
n(n==0) = NaN; % prevent divideByZero warnings
% Sum up non-NaNs, and divide by the number of non-NaNs.
m = sum(x) ./ n;