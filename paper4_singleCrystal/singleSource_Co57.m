%% ------------------ Co-57 bagged tree model ------------------
%% accuracy, Precision, recall and F1 score from classifiaction learner 

Co_data = readmatrix('/Users/harmonyteng/Desktop/paper4-ml/SC I-t ML/SC-Co-57-ML.csv');
Time    = Co_data(:,1);
Current = Co_data(:,2);
dIdt = [diff(Current) ./ diff(Time); NaN];

radiation_intervals_2 = [ ...
    301  433;
    729  855;
    1141 1273;
    1569 1691];

Label_num = zeros(size(Time));   % 0 = No Radiation

for i = 1:size(radiation_intervals_2,1)
    idx = Time >= radiation_intervals_2(i,1) & Time <= radiation_intervals_2(i,2);
    Label_num(idx) = 1;% 1 = Co-57
end

DataAll = [Time, Current, dIdt, Label_num];

% Remove NaN rows
validIdx = ~isnan(DataAll(:,3));
DataAll = DataAll(validIdx,:);

%% ------------------ MANUAL 80/20 SPLIT ------------------
rng(1);  % reproducibility
N = size(DataAll,1);
perm = randperm(N);

trainIdx = perm(1:round(0.8*N));
testIdx  = perm(round(0.8*N)+1:end);

TrainData = DataAll(trainIdx,:);
TestData  = DataAll(testIdx,:);

%% ------------------ CREATE TABLES ------------------
TrainTbl_Co = table( ...
    TrainData(:,2), ...  % Current
    TrainData(:,3), ...  % dI/dt
    categorical(TrainData(:,4), [0 1], ...
        {'Radiation off','Radiation on'}), ...
    'VariableNames', {'Current','dIdt','RadiationLabel'});

TestTbl_Co = table( ...
    TestData(:,2), ...
    TestData(:,3), ...
    categorical(TestData(:,4), [0 1], ...
        {'Radiation off','Radiation on'}), ...
    'VariableNames', {'Current','dIdt','RadiationLabel'});

%% ------------------ MODEL PREDICTION ------------------
features = TestTbl_Co(:,[1,2]);     
predicted_labels = trainedModel_Co.predictFcn(features);
true_labels = TestTbl_Co.RadiationLabel;

%% ------------------ PLOTS ------------------
figure;
tiledlayout(2,1,'TileSpacing','compact');

% ---- TRUE LABELS ----
nexttile;
gscatter(TestData(:,1), TestData(:,2), true_labels, 'br', 'ox');
xlabel('Time');
ylabel('Current');
title('True Radiation Status for Co-57 Radiation');
legend('Location','best');
grid on;

% ---- PREDICTED LABELS ----
nexttile;
gscatter(TestData(:,1), TestData(:,2), predicted_labels, 'br', 'ox');
xlabel('Time');
ylabel('Current');
title('Predicted Radiation Status for Co-57 Radiation');
legend('Location','best');
grid on;