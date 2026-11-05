%% MATH 582 - SVM Mini Project
% Training/Testing Split and Standardization
%
% Dataset: Breast Cancer Wisconsin
% Labels:
%   Malignant = +1
%   Benign    = -1

clear; clc; close all;

rng(582);  % Reproducibility

%% 1. Import Cleaned Dataset

filename = 'breast_cancer_clean.csv';

T = readtable(filename);

% Extract labels and features
y = T.Label;
T.Label = [];

X = table2array(T);
featureNames = T.Properties.VariableNames;

fprintf('\nCLEANED DATASET\n');
fprintf('-----------------------------\n');
fprintf('Observations: %d\n',size(X,1));
fprintf('Features: %d\n',size(X,2));

% Verify data
assert(all(ismember(y,[-1,1])), ...
    'Labels must be -1 or +1.');

assert(all(isfinite(X(:))), ...
    'Feature matrix contains invalid values.');

%% 2. Stratified Training/Testing Split

trainRatio = 0.80;

trainIdx = [];
testIdx = [];

classes = unique(y);

for i = 1:numel(classes)

    idx = find(y == classes(i));

    % Randomize observations within each class
    idx = idx(randperm(numel(idx)));

    % Number of training observations
    nTrain = round(trainRatio*numel(idx));

    % Assign training and testing observations
    trainIdx = [trainIdx; idx(1:nTrain)];
    testIdx = [testIdx; idx(nTrain+1:end)];

end

% Shuffle training and testing sets
trainIdx = trainIdx(randperm(numel(trainIdx)));
testIdx = testIdx(randperm(numel(testIdx)));

%% 3. Extract Training and Testing Data

X_train_raw = X(trainIdx,:);
X_test_raw = X(testIdx,:);

y_train = y(trainIdx);
y_test = y(testIdx);

%% 4. Feature Standardization

% Calculate mean and standard deviation
% using ONLY the training data.

mu = mean(X_train_raw,1);
sigma = std(X_train_raw,0,1);

% Prevent division by zero
sigma(sigma == 0) = 1;

% Standardize features
X_train = (X_train_raw - mu)./sigma;
X_test = (X_test_raw - mu)./sigma;

%% 5. Verify Standardization

fprintf('\nSTANDARDIZATION CHECK\n');
fprintf('-----------------------------\n');

fprintf('Maximum absolute training mean: %.6e\n', ...
    max(abs(mean(X_train,1))));

fprintf('Minimum training standard deviation: %.6f\n', ...
    min(std(X_train,0,1)));

fprintf('Maximum training standard deviation: %.6f\n', ...
    max(std(X_train,0,1)));

%% 6. Training/Testing Summary

fprintf('\nTRAINING/TESTING SUMMARY\n');
fprintf('-----------------------------\n');

fprintf('Training observations: %d\n',size(X_train,1));
fprintf('Testing observations: %d\n',size(X_test,1));

fprintf('\nTraining Class Distribution:\n');
fprintf('Malignant (+1): %d\n',sum(y_train==1));
fprintf('Benign (-1): %d\n',sum(y_train==-1));

fprintf('\nTesting Class Distribution:\n');
fprintf('Malignant (+1): %d\n',sum(y_test==1));
fprintf('Benign (-1): %d\n',sum(y_test==-1));

%% 7. Export Prepared Data

% Create output directory
outputDir = 'outputs';

if ~exist(outputDir,'dir')
    mkdir(outputDir);
end


%% 7. Export Prepared Data

% Create outputs folder in the same directory as this script
scriptDir = fileparts(mfilename('fullpath'));
outputDir = fullfile(scriptDir, 'outputs');

% Create folder if it does not exist
if ~isfolder(outputDir)
    mkdir(outputDir);
end

% Verify folder exists
assert(isfolder(outputDir), ...
    'Output directory could not be created.');

% Save MATLAB arrays
save(fullfile(outputDir,'breast_cancer_prepared.mat'), ...
    'X_train','X_test','y_train','y_test', ...
    'X_train_raw','X_test_raw', ...
    'mu','sigma','featureNames', ...
    'trainIdx','testIdx');

% Create tables with feature names
trainTable = array2table(X_train, ...
    'VariableNames',featureNames);
trainTable.Label = y_train;

testTable = array2table(X_test, ...
    'VariableNames',featureNames);
testTable.Label = y_test;

% Export CSV files
writetable(trainTable, ...
    fullfile(outputDir,'breast_cancer_train.csv'));

writetable(testTable, ...
    fullfile(outputDir,'breast_cancer_test.csv'));

% fprintf('\nFiles saved to:\n%s\n',outputDir);