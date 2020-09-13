# Fully Connected Feed-Forward Neural Network with Bayesian Optimized Hyperparameters for Predictions of Building Heating Loads

This project trains a fully connected feed-forward neural network to estimate the heating load on a building given eight different input features. The hyperparameters of the neural network are optimized using Bayesian Optimization.

![Neural Network](/Images/NN_model_sample.png)

---

## Folders and Files

This repo contains the following folders and files:

Folders:

* [Data](Data) : Raw data and description
  * ENB2012_data.csv - Raw data
  * ENB202_data.xlsx - Raw data in excel form
  * README.txt - Data description

* [Images](Images): Images used in the readme file

* [Models](Models): Results of Bayesian Optimization and Trained Model
  * Best_NN_Parameters.csv - Optimized Hyperparameters from Bayesian Optimization
  * NN_model.mat - MATLAB file of the trained neural network
  * NN_training_progress.csv - Training history of the optimal model.

Files:

* [Bldg_Energy_Efficiency_Bayesian_Opt.m](Bldg_Energy_Efficiency_Bayesian_Opt.m) - Bayesian Optimization of Hyperparameters and training of the neural network
* [Data_Exploration_Bldg_Energy_Efficiency.m](Data_Exploration_Bldg_Energy_Efficiency.m) - Data exploration
* [Bldg_Energy_Efficiency.m](Bldg_Energy_Efficiency.m) - Initial neural network building work. This was created before the Hyperparameter optimization script and was used to test the development.

---

## Dataset

The data for this project is from the UCI Machine Learning Repository titled [Energy Efficiency Data Set](https://archive.ics.uci.edu/ml/datasets/Energy+efficiency). It is composed of 768 samples of data created by Angeliki Xifara [[1]](#References) for buildings with different architecture characteristics in Athens, Greece. The dataset was also explored by Sadeghi et. al. [[2]](#References).

### Input Features

The input data contains 8 features:  

* X1 Relative Compactness (Unitless)
* X2 Surface Area (m^2)
* X3 Wall Area (m^2)
* X4 Roof Area (m^2)
* X5 Overall Height (m)
* X6 Orientation (Coded integers )
* X7 Glazing Area (Unitless)
* X8 Glazing Area Distribution (Coded integers)

The orientation (X6) is coded as (1) North, (2) South, (3) East, and (4) West facing. The Glazing Area (X8) is coded as follows: (0) no glazing, (1) uniform coating, with 25% glazing on each side, (2) North weighted with 55% glazing on the North side and 15% glazing on all other sides, (3) East weighted with 55% glazing on the East side and 15% glazing on all other sides, (4) South weighted with 55% glazing on the South side and 15% glazing on all other sides, and finally (5) West weighted with 55% glazing on the west side and 15% glazing on all the other sides [[2]](#References).

### Output Features

The two possible output features are the heating and cooling load as shown below. This analysis only considered the heating load.

* Y1 Heating Load (kWh/m^2)
* Y2 Cooling Load (kWh/m^2)

---

## Data Exploration

The first step in the analysis was to explore the data for any initial insights using the [Data_Exploration_Bldg_Energy_Efficiency.m](Data_Exploration_Bldg_Energy_Efficiency.m). This allowed the author to ensure that there were no missing values or obvious outliers in the data. Additionally, the a scatter plot matrix and simple linear model showed that X6 (orientation) had little effect on the target output of the heating load. Additionally, the feature X8, Glazing Area Distribution, was needed to be one-hot-encoded.

![Scatter Plot Matrix](/Images/Data_ScatterPlot_Matrix.png)

---

## Bayesian Optimization of the Hyperparameters

The next step was to develop the optimal neural network to predict the heating load (Y1). This was done using the Bayesian Optimization Function provided in the Matlab Statistics and Machine Learning Toolbox. The hyperparameters explored and their ranges are below. The objective function for the Bayesian optimization was to minimize the root mean squared error (RMSE) of the validation set (15% of the data also used for early stopping)

&nbsp;

| Variable | Description | Type | Range [min max] |
| ------ | --------- | ----- | --------- |
| FC_1 | Number of nodes in the first hidden layer | Integer | [12, 20] |
| FC_2 | Number of nodes in the second hidden layer | Integer | [7, 12] |
| FC_3 | Number of nodes in the third hidden layer | Integer | [2, 7] |
| num_layers_set | Number of hidden layers in the model | Integer | [1, 3] |
| InitialLearnRate | Initial learning rate of the neural net backpropagation optimizer | Continuous (on log scale) | [0.01, 1] |
| Momentum | Momentum for use only with stochastic gradient descent with momentum (sgdm) optimizer | Continuous | [0.8, 0.99] |
| solverName | Optimizer used during the backpropagation training of the neural net | Categorical | [sgdm, rmsprop, adam] |
| batch_size | Number of samples in each batch for stochastic gradient descent | Categorical | [4, 8, 16, 32, 64, 128, 256, 512] |
| L2Regularization | L2 Regularization factor used to multiply by the net weights and added to the loss function during backpropagation | Continuous (on log scale) | [1e-10, 1e-2] |

&nbsp;

These hyperparameter settings and their ranges were sampled and optimized by the MATLAB [Bayesian Optimization function](https://www.mathworks.com/help/stats/bayesopt.html#namevaluepairarguments). The inputs to the optimization process included 500 evaluation steps using the 'expected-improvement-plus' Acquisition Function. The result of the optimization process are shown below.

![Sample Output](/Images/Bayesian_Optimization_Progress.png)

Five other hyperparameters used in the neural network were not included in the Bayesian search. These were:

* Max Epochs: 500
* Gradient Threshold: 1
* Validation Patience: 5
* Learning Rate Drop Factor: 0.5
* Leaky Relu alpha: 0.1

The Bayesian Optimization took a little over an hour to run in single-core mode on a standard laptop. The results of the Bayesian Optimization showed the optimal combination of hyperparameters for the neural network:

* Number of hidden layers: 3
* Number of nodes in first hidden layer: 17
* Number of nodes in second hidden layer: 10
* Number of nodes in third hidden layer: 3
* Backpropagation Optimizer: adam
* Initial learning rate: 0.01024
* Batch Size: 8
* L2 Regularization Factor: 6.7307e-5

---

## Neural Net Model

The program used a fully-connected feed-forward neural network consisting of three hidden layers. The input layer had 12 nodes (X8 was converted to one-hot-encoding resulting in 7 original inputs X1-X7 and five inputs for the X8 one-hot-encoded vector), and output layer had one node. Each of the three hidden layers was followed by a [leaky rectified linear activation function](https://www.mathworks.com/help/deeplearning/ref/nnet.cnn.layer.leakyrelulayer.html) (LReLu). The neural network was then trained using the optimized hyperparameters on 75% of the data (training set). 15% of the data was for early stopping during training (validation set), and the final neural network was tested on 10% of the data which was withheld from all training (test set).

![Sample Output](/Images/Network.png)

---

## Results

The final neural network was able to predict the required heating load on average to within 1 kWh/m^2 for all three data sets (training, validation, and test). The max observed error was less than 3 kWh/m^2.

![Sample Output](/Images/fit_metrics.png)

An examination of the actual v. predicted and residual v. predicted plots show that the model has good accuracy across the range of heating load predictions. This is further confirmed through the histogram of the residuals which shows that the model is generally not biased to over or under predict.

![Sample Output](/Images/DNN_regression_Regression_Fit_Charts.png)

---

## MATLAB Libraries

This simulation was built using MATLAB 2020a with the following libraries:

* Statistics Toolbox
* Deep Learning Toolbox
* Optimization Toolbox
* Parallel Computing Toolbox

---

## References

[1]  Xifara,  Angeliki and Athanasios Tsanas, "Energy efficiency Data Set." UCI Machine Learning Repository, Center for Machine Learning and Intelligent Systems [https://archive.ics.uci.edu/ml/datasets/Energy+efficiency]

[2] Sadeghi, A., Roohollah, Y. S., Young, William A. II, & Weckman, G. R. (2020). An intelligent model to predict energy performances of residential buildings based on deep neural networks. Energies, 13(3), 571. doi: http://dx.doi.org.libproxy.nps.edu/10.3390/en13030571