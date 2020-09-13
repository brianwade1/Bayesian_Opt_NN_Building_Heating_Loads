%% Description and setup
%Input data from: 
%Data from the UIC Machine Learning Repository: Energy Efficiency Data Set
%https://archive.ics.uci.edu/ml/datasets/Energy+efficiency

%% Program setup
clear
clc
close all
start_time=tic; %Timer
rng(42)

%% User inputs
val_perc = 0.15; %percentage of data for validation set
test_perc = 0.1; %percentage of data for test test

data_folder = 'Data'; 
input_filename = 'ENB2012_data.csv'; % input file

%% Read input Data
% Read data
input = readtable(fullfile(data_folder,input_filename));

% Divide into input and target labels
x_end_col = size(input,2)-2; %input data
fit_col = size(input,2)-1; %Look at heating load

% Change data and label format to array
data = table2array(input(:,1:x_end_col));
target = table2array(input(:,fit_col));

%% Get data stats
% Check for NA  -- No NA found
[row_data_NA, col_data_NA] = find(isnan(data));
[row_target_NA, col_target_NA] = find(isnan(target));

% Summary of data
summary = summary(input);

%% Visualize the data
figure()
plotmatrix([data,target])
saveas(gcf,'Images/Data_ScatterPlot_Matrix.png')

% Convert col 8 (Glazing Area Distro) to one hot encoded
onehot = bsxfun(@eq, data(:,8), 1:max(data(:,8)));
data(:,8)=[];
data = [data,onehot];

% Fit linear model to check variable importance
mdl = fitlm(data, target);
tbl = anova(mdl)

% Remove X6 - Orientation from data
data(:,6) = [];

% Redo the model
mdl = fitlm(data, target);
tbl = anova(mdl)