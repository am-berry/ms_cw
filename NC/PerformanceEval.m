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
colsDrop = {'perimeter_mean', 'area_mean', 'concavity_mean',...
            'concavePoints_mean', 'perimeter_se', 'area_se',...
            'concavity_se', 'fractal_dimension_se','radius_worst',...
            'texture_worst', 'perimeter_worst', 'area_worst', ...
            'smoothness_worst', 'compactness_worst', 'concavity_worst',...
            'concavePoints_worst', 'fractal_dimension_worst'};

% Drop overly-correlated variables      
data = removevars(data,colsDrop);
y = data(:, 1);
data = normalize(data(:, 3:end));
data = [y data];
data = table2array(data);


%% Cross validation

indices = crossvalind('Kfold', data(:,1), 5);
svm_scores = [];
net_scores = [];

for i = 1:5
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
 
 
 
 
 % %% Building the Classifiers
% 
% % We will use the parameters which we obtained from the results of the
% % tuning
% 
% % ----------------------- SVM classifier ---------------------------------
% 
% tic;
% SVM_classifier = fitcsvm(X, y);
% SVM_time = toc;
% 
% SVM_cv = cvpartition(SVM_classifier.NumObservations, 'KFold', 5);
% SVM_cross_validated_model = crossval(SVM_classifier, 'cvpartition', SVM_cv);
% 
% tic;
% [SVM_predictions, SVM_scores] = kfoldPredict(SVM_cross_validated_model);
% SVM_pred_time = toc;
% 
% % ---------------------- Neural net --------------------------------------
%     
% tic;
% NN_classifier = fitnet(10);
% NN_classifier = train(NN_classifier, arr_X', arr_y');
% nn_time = toc;
% 
% NN_cv = cvpartition(length(arr_X), 'KFold', 5);
% NN_cv_model = crossval(NN_classifier, 'cvpartition', NN_cv);
% 
% tic;
% [NN_predictions, NN_scores] = kfoldPredict(NN_cv_model);
% nn_pred_time = toc;
% disp(nn_time);
% disp(nn_pred_time);
% 
% %% Visualise the Confusion Matrices
% 
% % Get the confusion matrix of the aggregated results of the 5 experiments
% 
% SVM_cm = confusionmat(arr_y, SVM_predictions);
% 
% % Plot the confusion matrices for both classifiers
% figure(1)
% plotconfusion(transpose(arr_y), transpose(SVM_predictions));
% title('SVM Confusion Matrix');
% 
% %% Calculate the Performance Metrics
% 
% [obs, col] = size(y);
% 
% %True Positives      | True Negatives      | False Positives     | False Negatives
% SVM_TP = SVM_cm(1,1);   SVM_TN = SVM_cm(2,2);   SVM_FP = SVM_cm(2,1);   SVM_FN = SVM_cm(1,2);
% 
% %Accuracy
% SVM_accuracy = (SVM_TP + SVM_TN)/   obs;
% 
% %Error
% SVM_error = (SVM_FP + SVM_FN)/obs;
% 
% %Precision
% SVM_prec = SVM_TP/(SVM_TP+SVM_FP);
% 
% %Recall
% SVM_recall = SVM_TP/(SVM_TP + SVM_FN);
% 
% %F1  score
% SVM_F1 = (2*SVM_prec*SVM_recall)/(SVM_prec + SVM_recall);
% 
% %Fowlkes–Mallows index
% % Determines the similarity between the two clusters obtained after the clustering algorithm
% SVM_FMI = sqrt(SVM_prec*SVM_recall);
% 
% %Plot the Receiver Operating Characteristic (ROC) curve
% [SVM_X,SVM_Y,SVM_T,SVM_AUC] = perfcurve(arr_y,SVM_scores(:,1),0);
% 
% figure(3)
% plot(SVM_X,SVM_Y,'LineWidth',1,'color','g')
% hold on
% title('ROC Curve of Classifiers');
% xlabel('False Positive Rate'); 
% ylabel('True Positive Rate');
% legend off
% 
% %% Print the calculated Performance Metrics
% 
% fprintf('EVALUATION OF THE PERFORMANCE OF THE CLASSIFIERS\n');
% fprintf('------------------------------------------------\n');
% fprintf('NAIVE BAYES PERFORMANCE \n');
% fprintf('Accuracy: %.3f\n', SVM_accuracy);
% fprintf('Error: %.3f\n', SVM_error);
% fprintf('Precision: %.3f\n', SVM_prec);
% fprintf('Recall: %.3f\n', SVM_recall);
% fprintf('F1 Score: %.3f\n', SVM_F1);
% fprintf('Fowlkes–Mallows index: %.3f\n', SVM_FMI);
% fprintf('Area Under ROC Curve: %.3f\n', SVM_AUC);
% fprintf('Generalisation Error: %.3f\n', kfoldLoss(SVM_cross_validated_model));
% fprintf('Time to build model: %.3f\n', SVM_time);
% fprintf('Prediction Time: %.3f\n', SVM_pred_time);
% fprintf('------------------------------------------------\n');
