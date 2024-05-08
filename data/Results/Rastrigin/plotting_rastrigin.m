figure
t = tiledlayout(1,2,'TileSpacing','Compact','Padding','Compact');

set(gcf,'color','w');
nexttile
loglog(bestrewardcma,LineWidth=2)
hold on
loglog(bestrewardpepg,LineWidth=2)
loglog(bestrewardpso,LineWidth=2)
loglog(bestrewardspsalocal,LineWidth=2)
loglog(bestrewardfd1000,LineWidth=2)
loglog(bestrewardfd1,LineWidth=2)

%ylim([1e-4 ,2*1e3])
grid on
ylabel('Score')
xlabel('Epochs')
set(gca,'Fontsize',16)
nexttile

loglog(linspace(0,time(1),length(bestrewardcma)),bestrewardcma,LineWidth=2)
hold on
loglog(linspace(0,time(2),length(bestrewardpepg)),bestrewardpepg,LineWidth=2)
loglog(linspace(0,time(3),length(bestrewardpso)),bestrewardpso,LineWidth=2)
loglog(linspace(0,time(4),length(bestrewardspsalocal)),bestrewardspsalocal,LineWidth=2)
loglog(linspace(0,time(5),length(bestrewardfd1000)),bestrewardfd1000,LineWidth=2)
loglog(linspace(0,time(6),length(bestrewardfd1)),bestrewardfd1,LineWidth=2)
%ylim([1e-4 ,2*1e3])
grid on
xlabel('Time')
set(gca,'Fontsize',16)

% nexttile
% pop_size = 100;
% loglog(linspace(1,epochs(1),length(bestrewardcma))*pop_size,bestrewardcma,LineWidth=2)
% hold on
% loglog(linspace(1,epochs(2),length(bestrewardpepg))*pop_size,bestrewardpepg,LineWidth=2)
% loglog(linspace(1,epochs(3),length(bestrewardpso))*pop_size,bestrewardpso,LineWidth=2)
% loglog(linspace(1,epochs(4),length(bestrewardspsalocal))*2,bestrewardspsalocal,LineWidth=2)
% loglog(linspace(1,epochs(5),length(bestrewardfd1000))*2*1000,bestrewardfd1000,LineWidth=2)
% loglog(linspace(1,epochs(6),length(bestrewardfd1))*2,bestrewardfd1,LineWidth=2)
% grid on
% %ylim([1e-4 ,2*1e3])
% 
% xlabel('# of function evals')
legend('CMAES','PEPG','PSO','SPSA','FD-1000','FD-1')

set(gca,'Fontsize',18)