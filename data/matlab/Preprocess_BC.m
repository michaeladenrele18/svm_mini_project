%% MATH 582 - SVM Mini Project
% Breast Cancer Dataset Cleaning
%
% Malignant = +1
% Benign    = -1

clear; clc; close all;

%% 1. Import Dataset

filename = 'Breast Cancer Dataset.csv';

opts = detectImportOptions(filename);
opts.VariableNamesLine = 1;
opts.DataLines = [2 Inf];

T = readtable(filename,opts);

fprintf('\nORIGINAL DATASET\n');
fprintf('-----------------------------\n');
fprintf('Observations: %d\n',height(T));
fprintf('Variables: %d\n',width(T));

disp(T(1:min(5,height(T)),:));

%% 2. Remove Unnecessary Columns

% Remove patient ID
if ismember('id',T.Properties.VariableNames)
    T.id = [];
end

% Remove columns containing only missing values
emptyCols = all(ismissing(T),1);
T(:,emptyCols) = [];

fprintf('\nAFTER REMOVING UNNECESSARY COLUMNS\n');
fprintf('Variables remaining: %d\n',width(T));

%% 3. Check Missing Values

fprintf('\nMISSING VALUE CHECK\n');
fprintf('-----------------------------\n');

missingPerColumn = sum(ismissing(T),1);

missingSummary = table( ...
    string(T.Properties.VariableNames)', ...
    missingPerColumn', ...
    'VariableNames',{'Variable','MissingCount'});

disp(missingSummary(missingSummary.MissingCount > 0,:));

if any(missingPerColumn > 0)
    error(['Missing values detected. Review the columns ' ...
           'before proceeding.']);
else
    fprintf('No missing values detected.\n');
end

%% 4. Check Duplicate Observations

fprintf('\nDUPLICATE CHECK\n');
fprintf('-----------------------------\n');

nBefore = height(T);

% Remove identical rows, keeping first occurrence
T = unique(T,'rows','stable');

nDuplicates = nBefore - height(T);

fprintf('Duplicate rows removed: %d\n',nDuplicates);

%% 5. Convert Diagnosis Labels

fprintf('\nLABEL ENCODING\n');
fprintf('-----------------------------\n');

diagnosis = string(T.diagnosis);

% Check for unexpected labels
if any(~ismember(diagnosis,["M","B"]))
    error('Unexpected diagnosis labels detected.');
end

% Malignant = +1, Benign = -1
y = -ones(height(T),1);
y(diagnosis == "M") = 1;

% Remove original diagnosis column
T.diagnosis = [];

%% 6. Extract Feature Matrix

X = table2array(T);
featureNames = T.Properties.VariableNames;

if ~isnumeric(X)
    error('Feature matrix contains nonnumeric variables.');
end

if any(~isfinite(X(:)))
    error('Feature matrix contains nonfinite values.');
end

%% 7. Dataset Summary

fprintf('\nCLEANED DATASET SUMMARY\n');
fprintf('-----------------------------\n');

fprintf('Observations: %d\n',size(X,1));
fprintf('Features: %d\n',size(X,2));

fprintf('Malignant (+1): %d\n',sum(y==1));
fprintf('Benign (-1): %d\n',sum(y==-1));

fprintf('Class balance: %.2f%% malignant\n', ...
    100*mean(y==1));

%% 8. Save Cleaned Data

% Save MATLAB arrays
save('breast_cancer_clean.mat', ...
    'X','y','featureNames');

% Save cleaned CSV with encoded labels
T_clean = array2table(X, ...
    'VariableNames',featureNames);

T_clean.Label = y;

writetable(T_clean,'breast_cancer_clean.csv');

fprintf('\nCleaned dataset saved successfully.\n');