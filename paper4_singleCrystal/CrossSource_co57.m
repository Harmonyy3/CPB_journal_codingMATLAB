%% ------------------ Train on Co-57 Test on Cs-137 Subspace model ------------------
%% accuracy, Precision, recall and F1 score from classifiaction learner 
%% ------------------ training data Co-57 ------------------
Co_data = readmatrix('/Users/harmonyteng/Desktop/paper4-ml/SC I-t ML/SC-Co-57-ML.csv');
Time    = Co_data(:,1);
Current = detrend (Co_data(:,2)); %%detrend 
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

%% ------------------ BUILD DATASET ------------------
DataAll = [Time, Current, dIdt, Label_num];

% Remove NaN rows
validIdx = ~isnan(DataAll(:,3));
DataAll = DataAll(validIdx,:);

%% ------------------ CREATE TABLES ------------------
Co57_trainTbl = table( ...
    DataAll(:,2), ...
    DataAll(:,3), ...
    categorical(DataAll(:,4), [0 1], ...
        {'Radiation off','Radiation on'}), ...
    'VariableNames', {'Current','dIdt','RadiationLabel'});


%% ----------------testing on Cs-137------------------
CsData = readmatrix('/Users/harmonyteng/Desktop/paper4-ml/SC I-t ML/new_CS.csv');
CsTime    = CsData(:,1);
CsCurrent = detrend (CsData(:,2)); %% detrend needed 

% dI/dt
CsdIdt = [diff(CsCurrent) ./ diff(CsTime); NaN];

Cs_intervals = [ ...
    253  384;
    656  782;
    1074 1214;
    1501 1627];

Label_num = zeros(size(CsTime));   % 0 = No Radiation

for i = 1:size(Cs_intervals,1)
    idx = CsTime >= Cs_intervals(i,1) & CsTime <= Cs_intervals(i,2);
    Label_num(idx) = 1;          % 1 = Cs-137 
end

Data_Cs = [CsTime, CsCurrent, CsdIdt, Label_num];

% Remove NaN rows
validIdx = ~isnan(Data_Cs(:,3));
Data_Cs = Data_Cs(validIdx,:);

%% testing 
Cs_TestTbl = table( ...
    Data_Cs(:,2), ...  % Current
    Data_Cs(:,3), ...  % dI/dt
    categorical(Data_Cs(:,4), [0 1], ...
        {'Radiation off','Radiation on'}), ...
    'VariableNames', {'Current','dIdt','RadiationLabel'});


%% ------------------ MODEL PREDICTION ------------------
features = Cs_TestTbl(:,[1,2]);     
predicted_labels = Cross_Co57.predictFcn(features);
true_labels = Cs_TestTbl.RadiationLabel;

%% ------------------ PLOTS ------------------
figure;
tiledlayout(2,1,'TileSpacing','compact');

% ---- TRUE LABELS ----
nexttile;
gscatter(Data_Cs(:,1), Data_Cs(:,2), true_labels, 'br', 'ox');
xlabel('Time');
ylabel('Current');
title('True Radiation Status for Cross-Source');
legend('Location','best');
grid on;

% ---- PREDICTED LABELS ----
nexttile;
gscatter(Data_Cs(:,1), Data_Cs(:,2), predicted_labels, 'br', 'ox');
xlabel('Time');
ylabel('Current');
title('Predicted Radiation Status for Cross-Source');
legend('Location','best');
grid on;

