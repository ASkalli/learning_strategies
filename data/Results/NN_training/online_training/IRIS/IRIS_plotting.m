figure; 

set(gcf,'color','w');

loglog(BPtestacc,LineWidth=2)
hold on
loglog(PEPGtestacc,LineWidth=2)
loglog(CMAtestacc,LineWidth=2)
loglog(PSOtestacc,LineWidth=2)
loglog(SPSAtestacc,LineWidth=2)
loglog(FD1testacc,LineWidth=2)

%ylim([50,100])

time_BP = 0.6;
time_CMA = 4;
time_PEPG = 5;
time_PSO = 3.5;
time_SPSA = 1.8;
time_FD = 3.7;



figure; 

% BPtestacc = [ BPtestacc];
% PEPGtestacc = [ PEPGtestacc];
% CMAtestacc = [10;10; CMAtestacc];
% PSOtestacc = [10;10; PSOtestacc];
% SPSAtestacc = [10;10; SPSAtestacc];
% FD1testacc = [10;10; FD1testacc];

%new focus detectors high bandwithds modulators DFB 

set(gcf,'color','w');

loglog(linspace(0,time_BP,length(BPtestacc)),BPtestacc,LineWidth=2)
hold on
loglog(linspace(0,time_PEPG,length(PEPGtestacc)),PEPGtestacc,LineWidth=2)
loglog(linspace(0,time_CMA,length(CMAtestacc)),CMAtestacc,LineWidth=2)
loglog(linspace(0,time_PSO,length(PSOtestacc)),PSOtestacc,LineWidth=2)
loglog(linspace(0,time_SPSA,length(SPSAtestacc)),SPSAtestacc,LineWidth=2)
loglog(linspace(0,time_FD,length(FD1testacc)),FD1testacc,LineWidth=2)

ylabel('Accuracy [%]')
xlabel('Time [s]')
legend('BP','PEPG','CMAES','PSO','SPSA','FD-1')

figure 

pop_size = [5 10 20 50 100 200 300 500];
n_params = 11274;
acc_vec = [86.81 91.23 95.55 97.82 97.96 97.21 98.15 98.11];

plot(100*pop_size/n_params,acc_vec,LineWidth=2)

ylabel('Accuracy [%]')
xlabel('Pop size ratio [%]')
