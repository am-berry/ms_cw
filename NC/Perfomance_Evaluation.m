% Evaluate the performance of each algorithm's best model through testing
% on the unseen data held out previously

% Clear Workspace 
clear all;

%% Importing the Dataset

% Loading in the train and test sets which were separated in the
% Parameter_Search.m file 

dir = sprintf('%s/train_features_reduced.csv', pwd);
opts = detectImportOptions(dir,'NumHeaderLines',0);
X_train = readtable(dir,opts);

dir = sprintf('%s/train_targets_reduced.csv', pwd);
opts = detectImportOptions(dir,'NumHeaderLines',0);
y_train = readtable(dir,opts);
dir = sprintf('%s/test_features_reduced.csv', pwd);
opts = detectImportOptions(dir,'NumHeaderLines',0);
X_test = readtable(dir,opts);

dir = sprintf('%s/test_targets_reduced.csv', pwd);
opts = detectImportOptions(dir,'NumHeaderLines',0);
y_test = readtable(dir,opts);

%% Light preprocessing

X_train = table2array(X_train);
y_train = table2array(y_train);
X_test = table2array(X_test);
y_test = table2array(y_test);

%% Best models

% We first retrain the best performing models of each type, then predict on the unseen X_test data,
% comparing to the y_test after

% Reduced dataset
% Neural network
rng('default');

tic;
net = fitnet([10,5]); 
net.trainFcn = 'trainbr';% bayesian regularisation backprop training
net.trainParam.lr = 0.05;
net.trainParam.epochs = 75;
net = train(net, X_train', y_train');
net_train_time = toc;

tic;
net_test_pred = net(X_test'); % test set preds
% network outputs probabilistic values, which we convert to binary through
% use of decision boundary of 0.5

for m = 1:length(net_test_pred)
    if net_test_pred(m) >= 0.5
        net_test_pred(m) = 1;
    else
        net_test_pred(m) = 0;
    end
end
net_test_time = toc;

% SVM
tic;
SVM = fitcsvm(X_train, y_train,...
                         'KernelFunction','linear',...
                         'BoxConstraint',1,...
                         'KernelScale',1);
svm_train_time = toc;

tic;
svm_test_pred = predict(SVM, X_test);
svm_test_time = toc;

%% Confusion matrices

figure(1);
plotconfusion(y_test', svm_test_pred')
title('Confusion matrix for SVM');

figure(2);
plotconfusion(y_test', net_test_pred)
title('Confusion matrix for neural network');

%% Final results on different metrics

[svm_acc, svm_recall, svm_spec, svm_prec, svm_f1, svm_fmi] = evaluation(svm_test_pred, y_test);
[net_acc, net_recall, net_spec, net_prec, net_f1, net_fmi] = evaluation(net_test_pred, y_test);

%% AUC

[svm_x,svm_y,svm_t,svm_auc] = perfcurve(y_test, svm_test_pred, 1);
[net_x, net_y, net_t, net_auc] = perfcurve(y_test, net_test_pred, 1);

figure(3);
plot(svm_x, svm_y);
hold on;
plot(net_x, net_y);
xlabel('False positive rate') ;
ylabel('True positive rate');
title('ROC for Classification');
legend('SVM ROC curve (AUC = 0.961)', 'Neural network ROC curve (AUC = 0.936)');

%% Output of final results 
fprintf('EVALUATION OF THE PERFORMANCE OF THE CLASSIFIERS\n');
fprintf('------------------------------------------------\n');
fprintf('SVM performance \n');
fprintf('Accuracy: %.3f\n', svm_acc);
fprintf('Error: %.3f\n', 1-svm_acc);
fprintf('Specificity: %.3f\n', svm_spec);
fprintf('Precision: %.3f\n', svm_prec);
fprintf('Recall: %.3f\n', svm_recall);
fprintf('F1 Score: %.3f\n', svm_f1);
fprintf('Fowlkes–Mallows index: %.3f\n', svm_fmi);
fprintf('Area Under ROC Curve: %.3f\n', svm_auc);
fprintf('Time to train model: %.3f\n', svm_train_time);
fprintf('Prediction Time: %.3f\n', svm_test_time);
fprintf('------------------------------------------------\n');
fprintf('Neural network performance \n');
fprintf('Accuracy: %.3f\n', net_acc);
fprintf('Error: %.3f\n', 1-net_acc);
fprintf('Specificity: %.3f\n', net_spec);
fprintf('Precision: %.3f\n', net_prec);
fprintf('Recall: %.3f\n', net_recall);
fprintf('F1 Score: %.3f\n', net_f1);
fprintf('Fowlkes–Mallows index: %.3f\n', net_fmi);
fprintf('Area Under ROC Curve: %.3f\n', net_auc);
fprintf('Time to build model: %.3f\n', net_train_time);
fprintf('Prediction Time: %.3f\n', net_test_time);
fprintf('------------------------------------------------\n');
