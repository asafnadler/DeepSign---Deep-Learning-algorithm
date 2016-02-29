from deep_learning import *
from reduction_functions import *
from optimizer import *
from deep_transformer import *
from keras.regularizers import *
import pickle


# This function use pickle to save objects into files.
# We use it to save the models created.
def save_object(obj, filename):
    with open(filename, 'wb') as output:
        pickle.dump(obj, output, pickle.HIGHEST_PROTOCOL)

# List of data sets paths
csv_sources = ["D:\\Datasets\\ailerons\\ailerons.csv",
               "D:\\Datasets\\amazon initial 20 30 10000\\Amazon_initial_50_30_10000.csv",
               "D:\\Datasets\\coil2000 - 2 classes, unbalanced\\coil2000.csv",
               "D:\\Datasets\\dbworld_bodies\\dbworld_bodies.csv",
               "D:\\Datasets\\dbworld_subjects\\dbworld_subjects.csv",
               "D:\\Datasets\\optdigits - 10 classes, balanced\\optdigits.csv",
               "D:\\Datasets\\slice_localization_data\\slice_localization_data.csv",
               "D:\\Datasets\\sonar - 2 classes, noisy features\\sonar.csv",
               "D:\\Datasets\\UJIndoorLoc\\trainingData.csv",
               "D:\\Datasets\\whight lifting\\Example_WearableComputing_weight_lifting_exercises_biceps_curl_variations.csv"]

# list of names of the responses classes for each data set according to the data sets list
classes = ["Goal",
           "class",
           "CARAVAN",
           "CLASS",
           "CLASS",
           "Class",
           "reference",
           "Type",
           "USERID",
           "classe"]

# list of counts to reduce to for each data set according to the data sets list
reduction_counts = [20,
                    100,
                    20,
                    50,
                    30,
                    20,
                    20,
                    20,
                    75,
                    20]

# Following are different parameters given to the Deep Transformer.
# We defined different kinds of parameters according to successes or failures of their formers.
# default params
params1 = {"activation": 'relu', "loss": 'mse', "optimizer": 'rmsprop',
           "nb_epoch": 100, "batch_size": 128, "reduction_function": half_reduction_function,
           "hidden_dropout": None, "input_dropout": None, "activity_regularizer": None, "denoise_ratio": None}
# deepsign params mild - as described in deepsign article but with less aggressive features
params2 = {"activation": 'relu', "loss": 'mse', "optimizer": deep_sign_sgd_optimizer,
           "nb_epoch": 100, "batch_size": 20, "reduction_function": deep_sign_reduction_function,
           "hidden_dropout": None, "input_dropout": None, "activity_regularizer": None, "denoise_ratio": None}
# deepsign params - as described in deepsign article with mild dropout and denoising
params3 = {"activation": 'relu', "loss": 'mse', "optimizer": deep_sign_sgd_optimizer,
           "nb_epoch": 200, "batch_size": 20, "reduction_function": deep_sign_reduction_function,
           "hidden_dropout": 0.1, "input_dropout": None, "activity_regularizer": activity_l2(), "denoise_ratio": 0.1}
# deepsign params harsh - as described in deepsign article
params4 = {"activation": 'relu', "loss": 'mse', "optimizer": deep_sign_sgd_optimizer,
           "nb_epoch": 1000, "batch_size": 20, "reduction_function": deep_sign_reduction_function,
           "hidden_dropout": 0.2, "input_dropout": None, "activity_regularizer": activity_l2(), "denoise_ratio": 0.2}
# Trying to improve params for our causes
params5 = {"activation": 'relu', "loss": 'mse', "optimizer": deep_sign_sgd_optimizer,
           "nb_epoch": 200, "batch_size": 128, "reduction_function": deep_sign_reduction_function,
           "hidden_dropout": 0.4, "input_dropout": None, "activity_regularizer": activity_l2(), "denoise_ratio": None}
# Trying to improve params for our causes
params6 = {"activation": 'softmax', "loss": 'mse', "optimizer": 'rmsprop',
           "nb_epoch": 100, "batch_size": 128, "reduction_function": half_reduction_function,
           "hidden_dropout": 0.4, "input_dropout": None, "activity_regularizer": None, "denoise_ratio": None}
# Trying to improve params for our causes
params7 = {"activation": 'relu', "loss": 'mse', "optimizer": "rmsprop",
           "nb_epoch": 100, "batch_size": 20, "reduction_function": gradient_reduction_function,
           "hidden_dropout": 0.5, "input_dropout": None, "activity_regularizer": activity_l2(), "denoise_ratio": None}
# Trying using 1 epoch
params8 = {"activation": 'relu', "loss": 'mse', "optimizer": "rmsprop",
           "nb_epoch": 1, "batch_size": 20, "reduction_function": gradient_reduction_function,
           "hidden_dropout": 0.5, "input_dropout": None, "activity_regularizer": activity_l2(), "denoise_ratio": None}

# List of all parameters to use on each of the data sets from data sets list
params = [params1, params2, params3, params4, params5, params6, params7]

# going over all data sets list
for i in range(0, len(csv_sources)):
    # going over all params
    for j in range(0, len(params)):
        # Create a deep transformer object
        deep_transformer = DeepTransformer()
        # assign the corresponding params
        deep_transformer.set_params(**(params[j]))
        # Run deep learning algorithm on it
        deep_learn_csv(csv_path=csv_sources[i], deep_transformer=deep_transformer,
                       result_file_path=csv_sources[i].replace(".csv", "_dl_" + repr(j + 1) + ".csv"),
                       responses=classes[i], target_feature_number=reduction_counts[i], verbose=1)
        # save model to file so it can be used later again
        save_object(deep_transformer, csv_sources[i].replace(".csv", "_dl_" + repr(j + 1) + ".pkl"))
