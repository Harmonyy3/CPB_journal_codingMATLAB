%% plot can be derived from classification learner 
%% CPB AND PMMA PLOTS

% Load Strain_y1
data_y1 = readtable('ddnA/work/harmony/pro1/strain_y1.csv');
time   = datetime(data_y1{:,1}, 'InputFormat','yyyy-MM-dd HH:mm:ss'); 
time_sec = seconds(time - time(1));   % convert datetime to elapsed seconds
strain = data_y1{:,2};
dstrain = [0; diff(strain) ./ diff(time_sec)];   % first element set to 0 to keep size consistent


radiation_intervals = [...
    datetime('2024-04-04 09:56:30', 'InputFormat','yyyy-MM-dd HH:mm:ss'), ...
    datetime('2024-04-04 10:07:43.5', 'InputFormat','yyyy-MM-dd HH:mm:ss.S'); 
    datetime('2024-04-04 11:32:07.9', 'InputFormat','yyyy-MM-dd HH:mm:ss.S'), ...
    datetime('2024-04-04 11:41:32.7', 'InputFormat','yyyy-MM-dd HH:mm:ss.S'); 
    datetime('2024-04-04 13:36:34.4', 'InputFormat','yyyy-MM-dd HH:mm:ss.S'), ...
    datetime('2024-04-04 13:47:52.5', 'InputFormat','yyyy-MM-dd HH:mm:ss.S'); 
    datetime('2024-04-04 14:10:39.5', 'InputFormat','yyyy-MM-dd HH:mm:ss.S'), ...
    datetime('2024-04-04 14:22:23.6', 'InputFormat','yyyy-MM-dd HH:mm:ss.S')];

radiation_status=zeros(size(time));

for i = 1:size(radiation_intervals,1)
    on_idx = time >= radiation_intervals(i,1) & time <= radiation_intervals(i,2);
    radiation_status(on_idx) = 1; % mark ON as 1
end

Strainy1 = table(time, strain, dstrain, radiation_status);

majority = Strainy1(Strainy1.radiation_status == 0, :);
minority = Strainy1(Strainy1.radiation_status == 1, :);

% Oversample minority class to match majority count
idx = randi(height(minority), height(majority), 1);
minority_oversampled = minority(idx, :);

% Combine and shuffle
balancedData = [majority; minority_oversampled];
balancedData = balancedData(randperm(height(balancedData)), :);

fprintf('Balanced dataset size: %d (%.1f%% class 1)\n', ...
    height(balancedData), mean(balancedData.radiation_status) * 100);

% --- Split into 80% training and 20% testing ------------------------------
rng(1);  % for reproducibility
New_combinations = balancedData(randperm(size(balancedData, 1)), :);

% Split 80% training, 20% testing
N = size(New_combinations, 1);
train_size = round(0.8 * N);

Training_data_split = New_combinations(1:train_size, :);
Testing_data_split  = New_combinations(train_size+1:end, :);

TrainTbl = table( ...
    Training_data_split.strain, ...
    Training_data_split.dstrain, ...
    categorical(Training_data_split.radiation_status, [0 1], ...
        {'Radiation off','Radiation on'}), ...
    'VariableNames', {'Current','dIdt','RadiationLabel'});


TestTbl = table( ...
    Testing_data_split.strain, ...
    Testing_data_split.dstrain, ...
    categorical(Testing_data_split.radiation_status, [0 1], ...
        {'Radiation off','Radiation on'}), ...
    'VariableNames', {'Current','dIdt','RadiationLabel'});


load ('trainedModel_FBG.mat');
%%%%%
%% ------------------ MODEL PREDICTION ------------------
features = TestTbl(:,[1,2]);     
predicted_labels = trainedModel_FBG.predictFcn(features);
true_labels = TestTbl.RadiationLabel;

%% ------------------ PLOTS ------------------
figure;
tiledlayout(2,1,'TileSpacing','compact');

% ---- TRUE LABELS ----
nexttile;
gscatter( ...
    Testing_data_split.time, ...
    Testing_data_split.strain, ...
    true_labels, ...
    'br', 'ox');
xlabel('Time');
ylabel('Current');
title('True Radiation Status for Cs-137 Radiation');
legend('Location','best');
grid on;

% ---- PREDICTED LABELS ----
nexttile;
gscatter( ...
    Testing_data_split.time, ...
    Testing_data_split.strain, ...
    predicted_labels, ...
    'br', 'ox');

xlabel('Time');
ylabel('Current');
title('Predicted Radiation Status for Cs-137 Radiation');
legend('Location','best');
grid on;

% ===== SAVE FIGURES =====

fname_pdf = '/ddnA/work/harmony/pro1/predicted_labels_strain1.pdf';
exportgraphics(fig, fname_pdf, 'ContentType','vector');
close(fig);
