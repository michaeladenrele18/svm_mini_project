%% MATH 582 - SVM Mini Project
% Synthetic Binary Classification Datasets
%
% Dataset 1: Linearly Separable
% Dataset 2: Overlapping Classes
% Dataset 3: Nonlinear Concentric Circles
%
% Labels: -1 and +1

clear; clc; close all;

rng(582);

%% 1. Parameters

N = 200;
n = N/2;
trainRatio = 0.80;

% Output directory
scriptDir = fileparts(mfilename('fullpath'));
outputDir = fullfile(scriptDir,'outputs','synthetic');

if ~isfolder(outputDir)
    mkdir(outputDir);
end

%% 2. Dataset 1: Linearly Separable
% Guaranteed separation by x1 = 0

X_pos = [1 + abs(randn(n,1)), randn(n,1)];
X_neg = [-1 - abs(randn(n,1)), randn(n,1)];

X1 = [X_pos; X_neg];
y1 = [ones(n,1); -ones(n,1)];

%% 3. Dataset 2: Overlapping Classes
% Gaussian distributions with overlap

X_pos = 1.25*randn(n,2) + [0.8,0.8];
X_neg = 1.25*randn(n,2) - [0.8,0.8];

X2 = [X_pos; X_neg];
y2 = [ones(n,1); -ones(n,1)];

%% 4. Dataset 3: Nonlinear Circles

theta1 = 2*pi*rand(n,1);
theta2 = 2*pi*rand(n,1);

r_inner = 1 + 0.15*randn(n,1);
r_outer = 3 + 0.20*randn(n,1);

X_pos = [r_inner.*cos(theta1), ...
         r_inner.*sin(theta1)];

X_neg = [r_outer.*cos(theta2), ...
         r_outer.*sin(theta2)];

X3 = [X_pos; X_neg];
y3 = [ones(n,1); -ones(n,1)];

%% 5. Organize Datasets

datasets = {X1,X2,X3};
labels = {y1,y2,y3};

names = {'linear_separable', ...
         'overlapping', ...
         'nonlinear_circles'};

titles = {'Linearly Separable', ...
          'Overlapping Classes', ...
          'Nonlinear Concentric Circles'};

%% 6. Generate Splits and Export Data

figure('Color','w','Position',[100 100 1200 400]);

for k = 1:3

    X = datasets{k};
    y = labels{k};

    % Stratified split
    trainIdx = [];
    testIdx = [];

    for c = [-1,1]

        idx = find(y == c);
        idx = idx(randperm(numel(idx)));

        nTrain = round(trainRatio*numel(idx));

        trainIdx = [trainIdx;idx(1:nTrain)];
        testIdx = [testIdx;idx(nTrain+1:end)];

    end

    % Shuffle
    trainIdx = trainIdx(randperm(numel(trainIdx)));
    testIdx = testIdx(randperm(numel(testIdx)));

    % Training/testing arrays
    X_train = X(trainIdx,:);
    y_train = y(trainIdx);

    X_test = X(testIdx,:);
    y_test = y(testIdx);

    % Export MATLAB data
    save(fullfile(outputDir,[names{k},'.mat']), ...
        'X_train','y_train','X_test','y_test');

    % Export CSV data
    trainTable = array2table(X_train, ...
        'VariableNames',{'Feature1','Feature2'});
    trainTable.Label = y_train;

    testTable = array2table(X_test, ...
        'VariableNames',{'Feature1','Feature2'});
    testTable.Label = y_test;

    writetable(trainTable, ...
        fullfile(outputDir,[names{k},'_train.csv']));

    writetable(testTable, ...
        fullfile(outputDir,[names{k},'_test.csv']));

    % Visualization
    subplot(1,3,k);

    scatter(X(y==-1,1),X(y==-1,2), ...
        30,'b','filled');
    hold on;

    scatter(X(y==1,1),X(y==1,2), ...
        30,'r','filled');

    xlabel('Feature 1');
    ylabel('Feature 2');
    title(titles{k});

    legend('Class -1','Class +1', ...
        'Location','best');

    grid on;
    axis equal;

    fprintf('%s: %d training, %d testing\n', ...
        names{k},numel(y_train),numel(y_test));

end

sgtitle('Synthetic Binary Classification Datasets');

exportgraphics(gcf, ...
    fullfile(outputDir,'synthetic_datasets.png'), ...
    'Resolution',300);

fprintf('\nAll synthetic datasets exported successfully.\n');
