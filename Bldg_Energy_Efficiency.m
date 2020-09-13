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

%Hyperparameters!
FC_1 = 7; %number of nodes in first hidden layer
FC_2 = 7; %number of nodes in second hidden layer
FC_3 = 3; %number of nodes in third hidden layer
maxEpochs = 100; %stopping criteria - max training epochs
InitialLearnRate = 0.01; %initial learn rate for optimizer
leaky_epsilon = 0.1; %leak rate for negative size of leaky Relu layers
LearnRateDropFactor = 0.5; %learning rate drop multiplier
GradientThreshold = 1; %gradient clipping threshold

batch_size = 32; % batch size during training
ValidationPatience = 5; %stopping criteria - number of epochs with 
                        %increasing validation loss rate


%% Read input Data
% Read data
input = readtable(fullfile(data_folder,input_filename));

% Divide into input and target labels
x_end_col = size(input,2)-2; %input data
fit_col = size(input,2)-1; %Look at heating load

% Change data and label format to array
data = table2array(input(:,1:x_end_col));
target = table2array(input(:,fit_col));



% Drop x6 (Orientation)
%data(:,6) = [];

%% Prepare data for modeling

% Transpose because MATLAB likes features as rows for neural nets
X = data';
t = target';

% Scale data to mean of 0 and 1 standard deviation for each feature
[x_scaled,PX] = mapstd(X);

onehot = bsxfun(@eq, data(:,8), 1:max(data(:,8)));
x_scaled(8,:)=[];
x_scaled = [x_scaled;onehot'];


% Divide data into train, validation, and test sets
[trainInd,valInd,testInd] = dividerand(size(X,2), ...
    (1 - val_perc - test_perc), val_perc, test_perc);

x_train = x_scaled(:,trainInd);
t_train = t(:,trainInd);
x_val = x_scaled(:,valInd);
t_val = t(:,valInd);
x_test = x_scaled(:,testInd);
t_test = t(:,testInd);


%% Build the Deep Neural Net (NN) model

% Get the number of samples in the training data.
nFeatures = size(x_train,1);

% Number of output features
nResponses = size(t_train,1);

% Number of samples in the train, validation, and test sets
nSamples = size(x_train,2);
nValSamples = size(x_val,2);
nTestSamples = size(x_test,2);

% Matlab uses an imageInputLayer for the input layer to the deep NN, this
% layer expects a 4D array (3D array for each sample and the 4th dimension
% as the number of samples in the dataset). Therefor, we need to reshape
% the train, validation,and test sets so that each sample is a 3D array of
% size [1 1 number_of_features] and the forth dimension is the number of
% samples in that dataset resulting in each dataset (train, validation, and
% test) needing to be a 4D array of size [1, 1, number_of_features,
% number_of_samples_in_that_set]
Xtrain = reshape(x_train, [1,1,nFeatures,nSamples]);
Xval = reshape(x_val, [1,1,nFeatures,nValSamples]);
Xtest = reshape(x_test, [1,1,nFeatures,nTestSamples]);

% Create the deep NN. This starts with an imageInputLayer as the data input
% layer. This is followed by a number of hidden layers. Each hidden layer
% uses a fullyConnectedLayer and an activation function. The
% fullyConnectedLayer multiplies the outputs of the pervious layer by a 
% weight, adds those weighted inputs, and applies a bias. It does not have
% an inherent activation function (besides the linear activation). The
% activation functions used below are all leaky rectified linear units
% (Leaky Relu). The final layer, regressionLayer, calculates the loss
% function for use in the back propagation process. In this case, the loss
% function is mean squared error (MSE).
layers = [ ...
    imageInputLayer([1 1 nFeatures],'Name','Input')
    fullyConnectedLayer(FC_1,'Name','FC1')
    leakyReluLayer(leaky_epsilon, 'Name', 'LReLu1')
    fullyConnectedLayer(FC_2,'Name','FC2')
    leakyReluLayer(leaky_epsilon,'Name', 'LReLu2')
    fullyConnectedLayer(FC_3,'Name','FC3')
    leakyReluLayer(leaky_epsilon,'Name', 'LReLu3')
    fullyConnectedLayer(nResponses, 'Name', 'FC_output')
    regressionLayer('Name', 'Output')];

% Put the layer array into a layer graph object
Network = layerGraph(layers);

% Plot and save the NN structure.
figure()
plot(Network)
saveas(gcf,'Images/Network.png')

%% Train the Deep Neural Net (NN) model
% Validation frequency is how often to check the loss of the validation set
% (expressed in iterations (samples)). The below 
validationFrequency = floor(size(t_train,2)/batch_size);
LearnRateDropPeriod = floor(maxEpochs/10);

options = trainingOptions('adam', ...
    'MaxEpochs',maxEpochs, ...
    'MiniBatchSize',batch_size, ...
    'InitialLearnRate',InitialLearnRate, ...
    'LearnRateSchedule','piecewise', ...
    'LearnRateDropPeriod',LearnRateDropPeriod, ...
    'LearnRateDropFactor',LearnRateDropFactor, ...
    'GradientThreshold',GradientThreshold, ...
    'Shuffle','every-epoch', ...
    'ValidationData',{Xval,t_val'}, ...
    'ValidationFrequency',validationFrequency, ...
    'ValidationPatience',ValidationPatience, ...
    'Verbose',1,...
    'VerboseFrequency',validationFrequency,...
    'Plots','none');

% Train the network and store the trained net and the training stats.
[net,info] = trainNetwork(Xtrain,t_train',layers,options);

% Save net and training stats
writetable(struct2table(info), 'Models/NN_training_progress.csv')
save('Models/NN_model','net')

% Predict and check results
y_hat_train = predict(net,Xtrain);
y_hat_val = predict(net,Xval);
y_hat_test = predict(net,Xtest);

% Calculate regression metrics: root mean squared error (RMSE), mean 
% absolute error (MAE), coefficient of determination (R^2), and max 
% observed error. Plot regression accuracy visuals: actual v predicted,
% residual v. predicted, and residual histogram.
DNN_results = results('DNN regression',y_hat_train,y_hat_val,y_hat_test,...
    t_train',t_val',t_test');

%% Finish program
%display total time to complete tasks
tElapsed = toc(start_time); 
hour=floor(tElapsed/3600);
tRemain = tElapsed - hour*3600;
min=floor(tRemain/60);
sec = tRemain - min*60;
 
disp(' ')
disp('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')
disp('Program Complete!!!!!')
disp('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')
disp(' ')

% Display regression accuracy metrics
disp(['Time to complete: ',num2str(hour),' hours, ',...
    num2str(min),' minutes, ',num2str(sec),' seconds'])


function [results]=results(model,y_train,y_val,y_test,t_train,t_val,t_test)

    % METRICS -------------------------------------------------------
    %Test performance of model:
    dataset = {'Train';'Validation';'Test'};
    results = table(dataset);

    % Find residuals
    res_train = y_train - t_train;
    res_val = y_val - t_val;
    res_test = y_test - t_test;

    % average of the labeled data (used later in R2 calcs)
    t_train_mean = mean(t_train);
    t_val_mean = mean(t_val);
    t_test_mean = mean(t_test);

    % Sum of squares of error and total for train, validation, and test
    % sets
    SSE_train = sum((res_train).^2);
    SST_train = sum((t_train - t_train_mean).^2);

    SSE_val = sum((res_val).^2);
    SST_val = sum((t_val - t_val_mean).^2);

    SSE_test = sum((res_test).^2);
    SST_test = sum((t_test - t_test_mean).^2);

    % Find coefficient of determination (R^2)
    R2_train = 1 - (SSE_train / SST_train);
    R2_val = 1 - (SSE_val / SST_val);
    R2_test = 1 - (SSE_test / SST_test);

    % store R2 in results table
    results.R2 = [R2_train; R2_val; R2_test];

    % Find root mean squared error (RMSE)
    RMSE_train = sqrt(mean(res_train.^2));
    RMSE_val = sqrt(mean(res_val.^2));
    RMSE_test = sqrt(mean(res_test.^2));

    % store RMSE in results table
    results.RMSE = [RMSE_train; RMSE_val; RMSE_test];

    % Find max absolute error (MAE)
    MAE_train = mean(abs(res_train));
    MAE_val = mean(abs(res_val));
    MAE_test = mean(abs(res_test));

    % store MAE in results table
    results.MAE = [MAE_train; MAE_val; MAE_test];

    % Find max observed error
    max_error_train = max(abs(res_train));
    max_error_val = max(abs(res_val));
    max_error_test = max(abs(res_test));

    % store max observed error in results table
    results.max_error = [max_error_train; max_error_val; max_error_test];

    disp(' ')
    disp('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')
    disp(['Fit results for model = ',model])
    disp('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')
    disp(' ')
    
    % print results table to console
    disp(results)


    % PLOTS ----------------------------------------------
    % Prepare 45 degree and horizontal line for later plots
    min_y = min(min(y_train),min(y_test));
    max_y = max(max(y_train),max(y_test));
    line = (min_y: .1 : max_y); % 45-degree line
    hline = zeros(numel(line)); % horizontal line

    % Make figure
    figure('Renderer', 'painters', 'Position', [10 10 1600 900])

    % Training Data ----
    % Training set fit statistics
    subplot(3,4,1)
    text(0.1,0.9,'Fit Statistics - Training Data','fontweight',...
        'bold','fontsize',10)
    text(0.3,0.7,['RMSE = ',num2str(RMSE_train)])
    text(0.3,0.5,['MAE = ',num2str(MAE_train)])
    text(0.3,0.3,['R2 = ',num2str(R2_train)])
    text(0.3,0.1,['Max Error = ',num2str(max_error_train)])
    set(gca,'visible','off')

    % Actual v. predicted for training set
    subplot(3,4,2)
    hold on
    plot(y_train,t_train,'x')
    plot(line,line,'-')
    hold off
    title('RSM Regression - Training Set')
    xlabel('Actual')
    ylabel('Predicted')
    xlim([min_y max_y])
    ylim([min_y max_y])

    % Residual v. predicted for training set
    subplot(3,4,3)
    hold on
    plot(y_train,res_train,'x')
    plot(line,hline)
    hold off
    title('Residuals - Training Set')
    xlabel('Predicted')
    ylabel('Residual')
    xlim([min_y max_y])

    % Residual histogram
    subplot(3,4,4)
    histogram(res_train)
    title('Residuals - Training Set')
    xlabel('Residual')

    % Validation Data -----
    % Validation set fit statistics
    subplot(3,4,5)
    text(0.1,0.9,'Fit Statistics - Validation Data','fontweight',...
        'bold','fontsize',10)
    text(0.3,0.7,['RMSE = ',num2str(RMSE_val)])
    text(0.3,0.5,['MAE = ',num2str(MAE_val)])
    text(0.3,0.3,['R2 = ',num2str(R2_val)])
    text(0.3,0.1,['Max Error = ',num2str(max_error_val)])
    set(gca,'visible','off')

    % Actual v. predicted for training set
    subplot(3,4,6)
    hold on
    plot(y_val,t_val,'x')
    plot(line,line,'-')
    hold off
    title('RSM Regression - Validation Set')
    xlabel('Actual')
    ylabel('Predicted')
    xlim([min_y max_y])
    ylim([min_y max_y])

    % Residual v. predicted for training set
    subplot(3,4,7)
    hold on
    plot(y_val,res_val,'x')
    plot(line,hline)
    hold off
    title('Residuals - Validation Set')
    xlabel('Predicted')
    ylabel('Residual')
    xlim([min_y max_y])

    % Residual histogram
    subplot(3,4,8)
    histogram(res_val)
    title('Residuals - Validation Set')
    xlabel('Residual')


    % Test data -----
    % Test set fit statistics
    subplot(3,4,9)
    text(0.15,0.9,'Fit Statistics - Test Data','fontweight',...
        'bold','fontsize',10)
    text(0.3,0.7,['RMSE = ',num2str(RMSE_test)])
    text(0.3,0.5,['MAE = ',num2str(MAE_test)])
    text(0.3,0.3,['R2 = ',num2str(R2_test)])
    text(0.3,0.1,['Max Error = ',num2str(max_error_test)])
    set(gca,'visible','off')

    % Actual v. predicted for test set
    subplot(3,4,10)
    hold on
    plot(y_test,t_test,'x')
    plot(line,line,'-')
    hold off
    title('RSM Regression - Test Set')
    xlabel('Actual')
    ylabel('Predicted')
    xlim([min_y max_y])
    ylim([min_y max_y])

    % Residual v. predicted for test set
    subplot(3,4,11)
    hold on
    plot(y_test,res_test,'x')
    plot(line,hline)
    hold off
    title('Residuals - Test Set')
    xlabel('Predicted')
    ylabel('Residual')
    xlim([min_y max_y])

    % Residual histogram for test set
    subplot(3,4,12)
    histogram(res_test)
    title('Residuals - Test Set')
    xlabel('Residual')

    % Save plots image to folder
    image_folder = 'Images';
    image_file = strcat(model,'_Regression_Fit_Charts.png');
    image_save_path = fullfile(image_folder,image_file);
    saveas(gcf, image_save_path)
end