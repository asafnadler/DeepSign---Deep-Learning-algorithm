# This file contains different kinds of optimizer that can be used in the deep learning algorithm
# To add an optimizer please add it here and import it from your file.
# Currently the file contains only one optimzer - which is the optimizer described in "deepsign" article

from keras.optimizers import *

deep_sign_sgd_optimizer = SGD(lr=0.001, decay=0.000001)
