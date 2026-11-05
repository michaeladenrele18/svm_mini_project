function results = evaluate_classifier(y_true, y_pred, plotFlag)
% MATH 582 - SVM Mini Project
% Binary Classifier Evaluation
%
% Inputs:
%   y_true   - Actual class labels (-1, +1)
%   y_pred   - Predicted class labels (-1, +1)
%   plotFlag - Generate confusion matrix (true/false)
%
% Output:
%   results  - Structure containing performance metrics

%% 1. Input Validation

if nargin < 3
    plotFlag = true;
end

y_true = y_true(:);
y_pred = y_pred(:);

assert(numel(y_true) == numel(y_pred), ...
    'Actual and predicted labels must have equal lengths.');

assert(~isempty(y_true), ...
    'Evaluation requires at least one observation.');

assert(all(ismember(y_true,[-1,1])) && ...
       all(ismember(y_pred,[-1,1])), ...
    'Labels must be -1 or +1.');

%% 2. Confusion Matrix Components

TP = sum(y_true == 1 & y_pred == 1);
TN = sum(y_true == -1 & y_pred == -1);
FP = sum(y_true == -1 & y_pred == 1);
FN = sum(y_true == 1 & y_pred == -1);

%% 3. Performance Metrics

accuracy = (TP + TN)/numel(y_true);

precision = safeDivide(TP,TP + FP);
recall = safeDivide(TP,TP + FN);
specificity = safeDivide(TN,TN + FP);

F1 = safeDivide(2*precision*recall,precision + recall);

%% 4. Store Results

results = struct();

results.Accuracy = accuracy;
results.Precision = precision;
results.Recall = recall;
results.Specificity = specificity;
results.F1 = F1;

results.TP = TP;
results.TN = TN;
results.FP = FP;
results.FN = FN;

results.ConfusionMatrix = [TN,FP;FN,TP];

%% 5. Display Results

fprintf('\nSVM CLASSIFICATION RESULTS\n');
fprintf('--------------------------------\n');
fprintf('Accuracy:    %.2f%%\n',100*accuracy);
fprintf('Precision:   %.2f%%\n',100*precision);
fprintf('Recall:      %.2f%%\n',100*recall);
fprintf('Specificity: %.2f%%\n',100*specificity);
fprintf('F1-score:    %.4f\n',F1);

fprintf('\nConfusion Matrix:\n');
disp(results.ConfusionMatrix);

%% 6. Plot Confusion Matrix

if plotFlag

    figure('Color','w');

    confusionchart(results.ConfusionMatrix, ...
        {'Negative (-1)','Positive (+1)'}, ...
        'Title','SVM Confusion Matrix', ...
        'RowSummary','row-normalized');

end

end

%% Local Function

function value = safeDivide(a,b)

if b == 0
    value = 0;
else
    value = a/b;
end

end
