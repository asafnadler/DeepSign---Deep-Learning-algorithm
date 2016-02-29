# DeepSign---Deep-Learning-algorithm
This repository contains Deep Learning algorithm developed in python using keras.

This code was developed as a part of a project in the course Applied Machine Learning in BGU university.
The code is based on the article deepsign by David, Omid E. and Nathan S. Netanyahu
Reference:
David, Omid E., and Nathan S. Netanyahu. "DeepSign: Deep learning for automatic malware signature generation and classification." Neural Networks (IJCNN), 2015 International Joint Conference on. IEEE, 2015.

Instruction for using the code:

The Transformer file - deepTransformer, contains the class DeepTransformer which extends TransformerMixin (sklearn)
The transformer can be used in order to fit data into reduce amount of features.
After fitting the transformer can be used for transforming data with the same structure into the reduced count of features.
The received data must be numerical!
The transformer receives data lists od list.
Unless desired - do not add the class column into the data, or else it will be compressed as well.

The Transformer should receive mode parameters which are set by the set_params method. You can get the parameters with get_params method.
Note that there are default parameters to all parameters:

reduction_function - is a function provided by the user which receives data and return a number of features to reduce to (see reduction function section) (default = half_reduction_function)
activation - a function which will be applied on the exit of every neuron in the network (default = 'relu')
loss - the function which according to it the algorithm will improve itself (default = 'mse')
optimizer - an object containing the learning rate and decay (default = 'rmsprop')
input_dropout - number of input neurons to drop on each round of the algorithm - used for reducing chance to overfitting (default = 0)
hidden_dropout - number of hidden neurons to drop on each round of the algorithm - used for reducing chance to overfitting (default = 0)
denoise_ratio - precentage of the data which will be noised (by zeroing the values) (default = 0)
batch_size - size of the batch on which the algorithm will work (default = 128)
nb_epoch - number of rounds on each layer of the algorithm (default = 100)
w_regularizer - weight regularizer (default = None)
b_regularizer - weight regularizer (default = None)
activity_regularizer - network regularizer (default = None)

The methods which should be used for data are - fit, transform and fir_transform

fit:  receives data x, target_feature_number and verbose:
fitting the model from the data structure x to reduce to target_feature_number count, note that if the reduction function is not correlated with the target, the number of features can be lower than the target but never higher.

transform: after fitting, it is possible to transform any data with the same structure as the one the model was trained on.
The function receive only the data and return the modified data

fit_transform: performs both fit and transforms the same data and return the transformed data.
Receives the same parameters as fit and return results as transform.


deep_learning file contains methods to work with the  transformer.
The main method deep_learn_csv receives csv path and a transformer and applies the transformer on the data, Then it transforms the data and saves it under a new csv file (path received as a parameter)
If the csv file is complicated it is possible to remove column (which will be added later in the printout) by providing the list of column names in remove_columns parameter. The same goes for ids column (ids parameter) and class column (responses parameter) if they exist.


The __init__ file contains the main program of the code.
The file contains 4 lists:
csv_sources : sources of the csv on which to operate.
classes : list of classes column in for each datasource (correlated to the csv_resource list)
reduction_counts : list of number of desired feature for each dataset (correlated to the csv_resource list)
params : list of parameters to use. the parameters are defined above it.

The main program goes over all csv resources and for each, it performs tranformation with each of the parameters.
The result will be |csv_resources|*|params| csv files which will be output to the same directory where the resource is with the extension _dl# to it (# according to the parameter order in list)



Instruction for contributing to the code:

It is possible to contribute different functions:

Optimizer_function file: should contain different types of optimizer for the deep_transformer, currently it present only one optimizer which is described in deepsign article.

reduction_function: contains different reduction function.
A reduction function is a function that receives adataset x, and returns a number to which the transformer should compress the feature count.
Mostly, the function looks at the length of the data (number of features is visible bt using len(x[0]) ) and then deciding the new count.
For examle half_reduction_function receives data with l parameters and returns the number l/2
Note that the number must be smaller then the number of current features, or else the algorithm will never stop.

Enjoy using and contributing to this code.
