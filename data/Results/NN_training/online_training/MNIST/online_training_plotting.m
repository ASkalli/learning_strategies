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
