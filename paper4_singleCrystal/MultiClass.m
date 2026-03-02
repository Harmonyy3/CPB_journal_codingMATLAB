Training_data = readmatrix('/Users/harmonyteng/Desktop/paper4-ml/SC I-t ML/new_CS.csv');
Training_time = Training_data(:,1);
Training_current =detrend (Training_data(:,2));
Training_didt=[diff(Training_current) ./ diff(Training_time); NaN];

Training_radiation_intervals = [253, 384; 656, 782; 1074, 1214; 1501, 1627];

Training_radiation_status = zeros(size(Training_time));

for i = 1:size(Training_radiation_intervals, 1)
    idx = (Training_time >= Training_radiation_intervals(i, 1)) & (Training_time <= Training_radiation_intervals(i, 2));
    Training_radiation_status(idx) = 1;  %%label 1 for Cs
end

train_data1 = [Training_time, Training_current, Training_didt, Training_radiation_status];


Training_data2 = readmatrix('/Users/harmonyteng/Desktop/paper4-ml/SC I-t ML/SC-Co-57-ML.csv');
time2 = Training_data2(:,1);
current2 = detrend (Training_data2(:,2));
didt2=[diff(current2) ./ diff(time2); NaN];

%%SC-Co-interval
radiation_intervals_2 = [301, 433; 729, 855; 1141, 1273; 1569, 1691];

radiation_status = zeros(size(time2));

for i = 1:size(radiation_intervals_2, 1)
    idx = (time2 >= radiation_intervals_2(i, 1)) & (time2 <= radiation_intervals_2(i, 2));
    radiation_status(idx) = 2; 
end
train_data2= [time2, current2, didt2, radiation_status];

combinations=[train_data1;train_data2];
combinations = combinations(~any(isnan(combinations),2), :);

rng(1);
New_combinations = combinations(randperm(size(combinations, 1)), :);

% Split 80% training, 20% testing
N = size(New_combinations, 1);
train_size = round(0.8 * N);

D_Training_data_split = New_combinations(1:train_size, :);
D_Testing_data_split  = New_combinations(train_size+1:end, :);
D_TrainingTable = table( ...
    D_Training_data_split(:,2), ...
    D_Training_data_split(:,3), ...
    categorical(D_Training_data_split(:,4), [0 1 2], ...
        {'Radiation off','Cs-137 Radiation','Co-57 Radiation'}), ...
    'VariableNames', {'Current','dIdt','RadiationLabel'});

D_TestingTable = table( ...
    D_Testing_data_split(:,2), ...
    D_Testing_data_split(:,3), ...
    categorical(D_Testing_data_split(:,4), [0 1 2], ...
        {'Radiation off','Cs-137 Radiation','Co-57 Radiation'}), ...
    'VariableNames', {'Current','dIdt','RadiationLabel'});


true_labels = D_TestingTable.RadiationLabel;
predicted_labels = Multiclass.predictFcn(D_TestingTable);

%%%
%% ------------------ PLOTS ------------------
figure;
tiledlayout(2,1,'TileSpacing','compact');

% ---- TRUE LABELS ----
nexttile;
gscatter(D_Testing_data_split(:,1), D_Testing_data_split(:,2), true_labels, 'bry', 'ox*');
xlabel('Time');
ylabel('Current');
title('True Radiation Status for Multi-Class Radiation');
legend('Location','best');
grid on;

% ---- PREDICTED LABELS ----
nexttile;
gscatter(D_Testing_data_split(:,1), D_Testing_data_split(:,2), predicted_labels, 'bry', 'ox*');
xlabel('Time');
ylabel('Current');
title('Predicted Radiation Status for Multi-Class Radiation');
legend('Location','best');
grid on;
