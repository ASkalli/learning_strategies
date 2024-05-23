figure; 

set(gcf,'color','w');

loglog(BPtestacc,LineWidth=2)
hold on
loglog(PEPGtestaccbest,LineWidth=2)
loglog(CMAtestacc,LineWidth=2)
loglog(PSOconvnet,LineWidth=2)
loglog(SPSAtestacc,LineWidth=2)
loglog(FD1testaccMNIST,LineWidth=2)

%ylim([50,100])

time_BP = 150;
time_CMA = 5040 * 6 ;
time_PEPG = 216;
time_PSO = 1200;
time_SPSA = 3780;
time_FD = 18060;



figure; 

BPtestacc = [10;10; BPtestacc];
PEPGtestaccbest = [10;10; PEPGtestaccbest];
CMAtestacc = [10;10; CMAtestacc];
PSOconvnet = [10;10; PSOconvnet];
SPSAtestacc = [10;10; SPSAtestacc];
FD1testaccMNIST = [10;10; FD1testaccMNIST];

%new focus detectors high bandwithds modulators DFB 

set(gcf,'color','w');

loglog(linspace(0,time_BP,length(BPtestacc)),BPtestacc,LineWidth=2)
hold on
loglog(linspace(0,time_PEPG,length(PEPGtestaccbest)),PEPGtestaccbest,LineWidth=2)
loglog(linspace(0,time_CMA,length(CMAtestacc)),CMAtestacc,LineWidth=2)
loglog(linspace(0,time_PSO,length(PSOconvnet)),PSOconvnet,LineWidth=2)
loglog(linspace(0,time_SPSA,length(SPSAtestacc)),SPSAtestacc,LineWidth=2)
loglog(linspace(0,time_FD,length(FD1testaccMNIST)),FD1testaccMNIST,LineWidth=2)

ylabel('Accuracy [%]')
xlabel('Time [s]')
legend('BP','PEPG','CMAES','PSO','SPSA','FD-1')

figure 

pop_size = [5 10 20 50 100 200 300 500];
n_params = 11274;

acc_vec_mnist = [86.81 91.23 95.55 97.82 97.96 97.7 98.15 98.11];
acc_vec_fashion = [75.86 78.84 80.93 84.58 84.91 86.61 86.62 86.71];

plot(100*pop_size/n_params,acc_vec_mnist,LineWidth=2)
hold on 
plot(100*pop_size/n_params,acc_vec_fashion,LineWidth=2)

ylabel('Accuracy [%]')
xlabel('Pop size ratio [%]')
