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

Instruction for contributing to the code:
