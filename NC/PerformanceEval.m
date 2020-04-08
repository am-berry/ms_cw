% In this script we are going to build the two classifiers using K-fold 
% cross-validation and produce important results for an effective comparison.

% Clear Workspace 
clear all;

%% Importing the Dataset

% Create a table of the Dataset
data_dir = sprintf('%s/data.csv', pwd);
opts = detectImportOptions(data_dir,'NumHeaderLines',0);
data = readtable(data_dir,opts);

%% Data Preprocessing

% Convert diagnosis to binary
% Replace ID (which we don't need) with the encoded diagnosis values
new_variable = cat2binary(data.diagnosis,{'M','B'},[1,0]);
data.id = new_variable;
data.Properties.VariableNames{1} = 'target';

% Variables that have high Pearson Correlation Coefficient (>0.8)
%%colsDrop = {'perimeter_mean', 'area_mean', 'concavity_mean',...
       %     'concavePoints_mean', 'perimeter_se', 'area_se',...
       %     'concavity_se', 'fractal_dimension_se','radius_worst',...
       %     'texture_worst', 'perimeter_worst', 'area_worst', ...
       %     'smoothness_worst', 'compactness_worst', 'concavity_worst',...
       %     'concavePoints_worst', 'fractal_dimension_worst'};

% Drop overly-correlated variables      
%data = removevars(data,colsDrop);
y = data(:, 1);
data = normalize(data(:, 3:end));
data = [y data];
data = table2array(data);


%% Cross validation

num_folds = 5;
indices = crossvalind('Kfold', data(:,1), num_folds);
svm_scores = [];
net_scores = [];

for i = 1:num_folds
    test_c = (indices == i);
    train_c = ~test_c;
    xtrain = data(train_c, 2:end);
    ytrain = data(train_c, 1);
    
    xtest = data(test_c, 2:end);
    ytest = data(test_c, 1);
    
    tic;
    svm = fitcsvm(xtrain, ytrain);
    svm_tt = toc;
    tic;
    ypred = predict(svm, xtest);
    svm_pt = toc;
    
    [acc, recall, spec, prec, f1, fmi] = evaluation(ytest, ypred);
    svm_scores = [svm_scores ; acc recall spec prec f1 fmi svm_tt, svm_pt];
    
    tic;
    net = fitnet(5);
    net = train(net, xtrain', ytrain');
    net_tt = toc;
    tic;
    ypred = net(xtest');
    net_pt = toc;
    for j = 1:length(ypred)
        if ypred(j) >= 0.5
            ypred(j) = 1;
        else
            ypred(j) = 0;
        end
    end
    [acc, recall, spec, prec, f1, fmi] = evaluation(ytest', ypred);
    net_scores = [net_scores ; acc recall spec prec f1 fmi net_tt, net_pt];

end

    
disp("SVM: ")
disp("Accuracy: " + mean(svm_scores(:,1)) + " Recall: " + mean(svm_scores(:, 2)) + " Specificity: " + mean(svm_scores(:,3)) + " Precision: " + mean(svm_scores(:,4)) + "  F1 score: " + mean(svm_scores(:, 5)) + " FMI: " + mean(svm_scores(:, 6)) + " Train time: " + mean(svm_scores(:, 7)) + " Test time: " + mean(svm_scores(:,8)))

disp("Neural networks: ")
disp("Accuracy: " + mean(net_scores(:,1)) + " Recall: " + mean(net_scores(:, 2)) + " Specificity: " + mean(net_scores(:,3)) + " Precision: " + mean(net_scores(:,4)) + "  F1 score: " + mean(net_scores(:, 5)) + " FMI: " + mean(net_scores(:, 6)) + " Train time: " + mean(net_scores(:, 7)) + " Test time: " + mean(net_scores(:,8)))
 