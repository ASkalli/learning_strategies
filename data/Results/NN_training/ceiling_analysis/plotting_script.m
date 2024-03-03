% Create a figure window
figure;
t = tiledlayout(2,2,'TileSpacing','Compact','Padding','Compact');

set(gcf,'color','w');
n_neurons_vec = [10,20,50,100,200,500,1000,2000];

% First subplot
%subplot(2, 2, 1); % (3 rows, 1 column, 1st subplot)
nexttile
plot(testacccompare',LineWidth=2)
xlabel('Minibatches');      % X-axis label
ylabel('Accuracy [%]'); % Y-axis label

% Second subplot
%subplot(2, 2, 2); % (3 rows, 1 column, 2nd subplot)
nexttile
semilogx(n_neurons_vec, max(testaccFFNN,[],2),'--.',LineWidth=2,MarkerSize=25);      % Plotting the FFNN model
hold on 
semilogx(n_neurons_vec, max(testaccELM,[],2),'--.',LineWidth=2,MarkerSize=25);      % Plotting the ELM model 

max_acc_lin = max(testacccompare,[],2);
max_acc_lin = max_acc_lin(3);

semilogx(n_neurons_vec, ones(size(n_neurons_vec))*max_acc_lin,'--',LineWidth=2);      % Plotting the linear model

xlabel('# of neurons');      % X-axis label
ylabel('Accuracy [%]'); % Y-axis label

% Third subplot
%subplot(2,2, 3); % (3 rows, 1 column, 3rd subplot)
nexttile
plot(e01, nan1,'--.',LineWidth=2,MarkerSize=25);      % Plotting the third set of data
xlabel('Accuracy [%]');       % X-axis label
ylabel('ELM neurons / FFNN neurons'); % Y-axis label



% Fourth subplot
%subplot(2, 2, 4); % (3 rows, 1 column, 3rd subplot)
nexttile
plot(quantization(1,:), quantization(2,:),'--.',LineWidth=2,MarkerSize=25);      % Plotting the third set of data
xlabel('# of bits');       % X-axis label
ylabel('Accuracy [%]'); % Y-axis label