%% MATH 582 - SVM Mini Project
% Test Classification Evaluation

clear; clc; close all;

%% 1. Load Prepared Breast Cancer Data

scriptDir = fileparts(mfilename('fullpath'));

dataFile = fullfile(scriptDir,'outputs', ...
    'breast_cancer_prepared.mat');

load(dataFile, ...
    'X_train','y_train','X_test','y_test');

%% 2. Train Reference Linear SVM

C = 1;

tic;

model = fitcsvm(X_train,y_train, ...
    'KernelFunction','linear', ...
    'BoxConstraint',C, ...
    'Standardize',false);

trainingTime = toc;

%% 3. Generate Predictions

y_train_pred = predict(model,X_train);
y_test_pred = predict(model,X_test);

%% 4. Evaluate Training Performance

fprintf('\nTRAINING PERFORMANCE\n');

trainResults = evaluate_classifier( ...
    y_train,y_train_pred,false);

%% 5. Evaluate Testing Performance

fprintf('\nTESTING PERFORMANCE\n');

testResults = evaluate_classifier( ...
    y_test,y_test_pred,true);

%% 6. Additional Model Information

nSupportVectors = size(model.SupportVectors,1);

fprintf('\nMODEL INFORMATION\n');
fprintf('--------------------------------\n');
fprintf('Kernel: Linear\n');
fprintf('C: %.2f\n',C);
fprintf('Support vectors: %d\n',nSupportVectors);
fprintf('Training time: %.6f seconds\n',trainingTime);

%% 7. Export Results

outputDir = fullfile(scriptDir,'outputs');

save(fullfile(outputDir,'reference_svm_results.mat'), ...
    'trainResults','testResults', ...
    'trainingTime','nSupportVectors','C');

fprintf('\nReference SVM evaluation complete.\n');
