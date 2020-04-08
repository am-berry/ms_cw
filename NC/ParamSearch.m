% Clear Workspace 
clear all;

%% Importing the Dataset

% Create a table of the Dataset
data_dir = sprintf('%s/data.csv', pwd);
opts = detectImportOptions(data_dir,'NumHeaderLines',0);
data = readtable(data_dir,opts);

%% Data Preprocessing

% Convert diagnosis to binary
% Sort the columns a bit, remove irrelevant ID column
new_variable = cat2binary(data.diagnosis,{'M','B'},[1,0]);
data.id = new_variable;
data.Properties.VariableNames{1} = 'target';

%colsDrop = {'perimeter_mean', 'area_mean', 'concavity_mean',...
 %           'concavePoints_mean', 'perimeter_se', 'area_se',...
  %          'concavity_se', 'fractal_dimension_se','radius_worst',...
  %          'texture_worst', 'perimeter_worst', 'area_worst', ...
  %          'smoothness_worst', 'compactness_worst', 'concavity_worst',...
  %          'concavePoints_worst', 'fractal_dimension_worst'};
        
%data = removevars(data,colsDrop);

% Cross validation (train: 90%, test: 10%) because our dataset is
% relatively small
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

%% Perform Hyperparameter Tuning

% Tuning over the three types of kernel function
KernelFunction = ["linear", "rbf", "polynomial"];

% Tuning over the Gamma and Box constraint hyperparameters
BoxConstraint = linspace(1,20,20);

Gamma = logspace(-2, 1, 4); % values of 0.01, 0.1, 1, 10

n = 0; % counter 
svm_error = zeros(1,240); % storing accuracy in an array

idx = randperm(numel(y_train), size(y_train, 1));
x = X_train(idx, :);
y = y_train(idx, :);
cv = cvpartition(y_train(idx), 'KFold', 5, 'Stratify', true);

% Iterate over the three types of function
for i=1:length(KernelFunction)
    % Iterate over the list of box constraints
    for j=1:length(BoxConstraint)
        % Iterate over the possible gammas
        for k=1:length(Gamma)
            %5 fold cross validation 
            avg_train_acc = 0;
            avg_test_acc = 0;
            for l = 1:5
                SVM = fitcsvm(x(cv.training(l), :), y(cv.training(l)),...
                         'KernelFunction',KernelFunction(i),...
                         'BoxConstraint',BoxConstraint(j),...
                         'KernelScale',Gamma(k));
                svm_train_pred = predict(SVM, x(cv.training(l), :));
                svm_test_pred  = predict(SVM, x(cv.test(l), :));
                [acc, recall, spec, prec, f1, fmi] = evaluation(svm_train_pred, y(cv.training(l)));
                [acc1, recall1, spec1, prec1, f11, fmi1] = evaluation(svm_test_pred, y(cv.test(l)));
                avg_train_acc = avg_train_acc + acc;
                avg_test_acc = avg_test_acc + acc1;
            end
            n = n+1;
            svm_error(1,n) = avg_test_acc / 5; %append to the error matrix
        end
    end                 
end

% Using the OptimizeHyperparameters function in the fitcsvm method
model = fitcsvm(X_train, y_train, 'OptimizeHyperparameters', 'auto', ...
    'HyperparameterOptimizationOptions', struct('AcquisitionFunctionName',...
    'expected-improvement-plus'));

%%
% Plot the Classification error against the Lambda values for the two
% penalty types
figure(1);
% Use the standard error as the value for the vertical error bars which is
% calculated by std/sqrt(number of observations)
%errorbar(BoxConstraint,svm_error(:,1:20),ones(size(svm_error(:,1:20)))*std(svm_error)/sqrt(length(BoxConstraint)));
%hold on
%errorbar(BoxConstraint,svm_error(:,21:end),ones(size(svm_error(:,1:20)))*std(svm_error)/sqrt(length(BoxConstraint)));
[X,Y] = meshgrid(BoxConstraint, Gamma);
Z = reshape(svm_error(:, 1:80), 4, 20);
mesh(X,Y,Z);
hold on
%surf(BoxConstraint, Gamma, svm_error(:, 1:80), 'o');


title('Classification Error Vs Lambda for the two Penalty Types')
zlim([0 1]);
xlabel('Box Constraint');
ylabel('Gamma');
zlabel('Classification Error');
