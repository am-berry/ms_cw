% Clear Workspace 
clear all;

%% Importing the Dataset

% Create a table of the Dataset
data_dir = sprintf('%s/data.csv', pwd);
opts = detectImportOptions(data_dir,'NumHeaderLines',0);
data = readtable(data_dir,opts);

%% Data Preprocessing

% Convert diagnosis to binary (0,1)
% Rearranging columns and removing the ID column
new_variable = cat2binary(data.diagnosis,{'M','B'},[1,0]);
data.id = new_variable;
data.Properties.VariableNames{1} = 'target';

% features to remove from our initial analysis
colsDrop = {'perimeter_mean', 'area_mean', 'concavity_mean',...
            'concavePoints_mean', 'perimeter_se', 'area_se',...
            'concavity_se', 'fractal_dimension_se','radius_worst',...
            'texture_worst', 'perimeter_worst', 'area_worst', ...
            'smoothness_worst', 'compactness_worst', 'concavity_worst',...
            'concavePoints_worst', 'fractal_dimension_worst'};
        
data = removevars(data,colsDrop);

% Extracting our holdout set for later model comparison
rng('default');
cv = cvpartition(size(data,1),'HoldOut',0.1);
idx = cv.test;

data_Train = data(~idx,:);
data_Test  = data(idx,:);

[rows, cols] = size(data); 

% Center and scale to have mean 0 and standard deviation 1
X_train = normalize(data_Train(:, 3:cols));
y_train = data_Train(:, 1);
X_test = normalize(data_Test(:, 3:cols));
y_test = data_Test(:, 1);
X_train = table2array(X_train);
y_train = table2array(y_train);
X_test = table2array(X_test);
y_test = table2array(y_test);

%% Set up k-fold cross validation partitions

num_folds = 4;
idx = randperm(numel(y_train), size(y_train, 1));
x = X_train(idx, :);
y = y_train(idx, :);
cv = cvpartition(y_train(idx), 'KFold', num_folds, 'Stratify', true);


%% Grid search for SVM

% Tuning over the three types of kernel function
KernelFunction = ["linear", "rbf", "polynomial"];

% Tuning over the Gamma and Box constraint hyperparameters
BoxConstraint = [1, 5, 10];

Gamma = logspace(-3, 0, 4); % values of 0.001, 0.01, 0.1, 1.0

n = 0; % counter
% preallocating arrays for storing errors 
train_svm_error = zeros(36,num_folds);
test_svm_error = zeros(36,num_folds); 

tic;
% Iterate over the three types of function
for i=1:length(KernelFunction)
    % Iterate over the list of box constraints
    for j=1:length(BoxConstraint)
        % Iterate over the possible gammas
        for k=1:length(Gamma)
            avg_train_acc = zeros(1, num_folds);
            avg_test_acc = zeros(1, num_folds);
            for l = 1:num_folds
                % k fold cross val, 4 folds
                SVM = fitcsvm(x(cv.training(l), :), y(cv.training(l)),...
                         'KernelFunction',KernelFunction(i),...
                         'BoxConstraint',BoxConstraint(j),...
                         'KernelScale',Gamma(k));
                svm_train_pred = predict(SVM, x(cv.training(l), :));
                svm_test_pred  = predict(SVM, x(cv.test(l), :));
                [acc, recall, spec, prec, f1, fmi] = evaluation(svm_train_pred, y(cv.training(l)));
                [acc1, recall1, spec1, prec1, f11, fmi1] = evaluation(svm_test_pred, y(cv.test(l)));
                avg_train_acc(1,l) = acc;
                avg_test_acc(1,l) = acc1;
            end
            n = n+1;
            % 
            train_svm_error(n,:) = avg_train_acc;
            test_svm_error(n, :) = avg_test_acc; 
        end
    end                 
end

time = toc;
disp(time);

%% Hyperparameter search for MLP

% sizes of the hidden layers 
first_hl_size = [5, 10, 15];
second_hl_size = [5, 10, 15];

% learning rates to parameterise
lrs = [0.01, 0.05, 0.1, 0.5];

n=0;
% preallocating arrays for storing errors
train_net_error = zeros(36, num_folds); 
test_net_error = zeros(36, num_folds);

tic;
for i=1:length(first_hl_size)
    for j=1:length(second_hl_size)
     % loop over the different first and second hidden layer sizes
        for k=1:length(lrs)
         % loop over the different learning rates 
            avg_train_acc = zeros(1,num_folds);
            avg_test_acc = zeros(1, num_folds);
            for l=1:num_folds
            % k fold cross validation, training each nn on a different fold
                net = fitnet([first_hl_size(i), second_hl_size(j)]); 
                net.trainFcn = 'trainbr';% bayesian regularisation backprop training
                net.trainParam.lr = lrs(k);
                net.trainParam.epochs = 75; % reduced to 75 epochs to avoid overfit
                net = train(net, x(cv.training(l),:)', y(cv.training(l))');
                net_train_pred = net(x(cv.training(l),:)'); % train set preds 
                net_test_pred = net(x(cv.test(l),:)'); % test set preds
                % decision threshold 0.5
                for m = 1:length(net_train_pred)
                    if net_train_pred(m) >= 0.5
                        net_train_pred(m) = 1;
                    else
                        net_train_pred(m) = 0;
                    end
                end
             
                for m = 1:length(net_test_pred)
                    if net_test_pred(m) >= 0.5
                        net_test_pred(m) = 1;
                    else
                        net_test_pred(m) = 0;
                    end
                end
            
                [acc, recall, spec, prec, f1, fmi] = evaluation(net_train_pred, y(cv.training(l)));
                [acc1, recall1, spec1, prec1, f11, fmi1] = evaluation(net_test_pred, y(cv.test(l)));
                % evaluating accuracy (and other metrics) on the test preds 
                avg_train_acc(1,l) = acc;
                avg_test_acc(1,l) = acc1;
            end
            n=n+1;
            train_net_error(n, :) = avg_train_acc; % storing avg of the train fold preds
            test_net_error(n, :) = avg_test_acc; % storing avg of the test fold preds
        end 
    end
end
time=toc;
disp(time);

%% write pertinent data to csv for later use 

writematrix(X_test, 'test_features_reduced.csv');
writematrix(y_test, 'test_targets_reduced.csv');
writematrix(X_train, 'train_features_reduced.csv');
writematrix(y_train, 'train_targets_reduced.csv');
%writematrix(train_net_error, 'net_train_errors_reduced.csv');
%writematrix(test_net_error, 'net_test_errors_reduced.csv');
%writematrix(train_svm_error, 'svm_train_errors_reduced.csv');
%writematrix(test_svm_error, 'svm_test_errors_reduced.csv');
