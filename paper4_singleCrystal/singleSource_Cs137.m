%% ------------------ Cs-137 data bagged tree model ------------------
%% accuracy, Precision, recall and F1 score from classifiaction learner 
Data = readmatrix('/Users/harmonyteng/Desktop/paper3&paper4_ML/paper4_singleCrystal/new_CS.csv');
Time    = Data(:,1);
Current = Data(:,2);

% dI/dt
dIdt = [diff(Current) ./ diff(Time); NaN];

Cs_intervals = [ ...
    253  384;
    656  782;
    1074 1214;
    1501 1627];

Label_num = zeros(size(Time));   % 0 = No Radiation

for i = 1:size(Cs_intervals,1)
    idx = Time >= Cs_intervals(i,1) & Time <= Cs_intervals(i,2);
    Label_num(idx) = 1;          % 1 = Cs-137
end

DataAll = [Time, Current, dIdt, Label_num];

% Remove NaN rows
validIdx = ~isnan(DataAll(:,3));
DataAll = DataAll(validIdx,:);

%% ------------------ MANUAL 80 / 20 SPLIT ------------------
rng(1);  % reproducibility
N = size(DataAll,1);
perm = randperm(N);

trainIdx = perm(1:round(0.8*N));
testIdx  = perm(round(0.8*N)+1:end);

TrainData = DataAll(trainIdx,:);
TestData  = DataAll(testIdx,:);

%% ------------------ table ------------------
TrainTbl = table( ...
    TrainData(:,2), ...  % Current
    TrainData(:,3), ...  % dI/dt
    categorical(TrainData(:,4), [0 1], ...
        {'Radiation off','Radiation on'}), ...
    'VariableNames', {'Current','dIdt','RadiationLabel'});

TestTbl = table( ...
    TestData(:,2), ...
    TestData(:,3), ...
    categorical(TestData(:,4), [0 1], ...
        {'Radiation off','Radiation on'}), ...
    'VariableNames', {'Current','dIdt','RadiationLabel'});

true_labels = TestTbl.RadiationLabel;

predicted_labels = trainedModelCs.predictFcn(TestTbl);

%%plot
figure;
tiledlayout(2,1,'TileSpacing','compact');

% ---- TRUE LABELS ----
nexttile; hold on;
gscatter(TestData(:,1), TestData(:,2), true_labels, 'br', 'ox');
xlabel('Time');
ylabel('Current');
title('True Radiation Status for Cs-137');
legend('Location','best');
grid on;

% ---- PREDICTED LABELS ----
nexttile; hold on;
gscatter(TestData(:,1), TestData(:,2), predicted_labels, 'br', 'ox');
xlabel('Time');
ylabel('Current');
title('Predicted Radiation Status for Cs-137');
legend('Location','best');
grid on;

