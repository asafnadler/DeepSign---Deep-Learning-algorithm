from deep_learning import *
from reduction_functions import *
from optimizer import *
from deep_transformer import *
from keras.regularizers import l2, activity_l2
import pickle
from sklearn.feature_extraction.text import CountVectorizer
import glob


# This function use pickle to save objects into files.
# We use it to save the models created.
def save_object(obj, filename):
    with open(filename, 'wb') as output:
        pickle.dump(obj, output, pickle.HIGHEST_PROTOCOL)

# Create a deep transformer object
deep_transformer = DeepTransformer()
# assign the corresponding params
deep_transformer.set_params(**({"activation": 'relu', "loss": 'mse', "optimizer": 'rmsprop',
   "nb_epoch": 100, "batch_size": 128, "reduction_function": half_reduction_function,
   "hidden_dropout": None, "input_dropout": None, "activity_regularizer": None, "denoise_ratio": None}))

# Run deep learning algorithm on it
data_path = r'../datasets/*.log.*'
print "Searching for data in {}".format(data_path)
fnames = [fname for fname in glob.glob(data_path)]
print "Using the following as a corpus : {}".format(fnames)

print "Replacing data with a unigram vector"
unigram_vectorizer = CountVectorizer("filename", analyzer='word', max_features=5000, binary=True, decode_error='ignore')
vectors = unigram_vectorizer.fit_transform(fnames).todense()
print "Vocabulary : {}".format(unigram_vectorizer.vocabulary_)
print "Unigram Vectorizer : {}".format(unigram_vectorizer)

print "Vectors : {}".format(vectors)

print "Applying deep learning... (may take a while)"
deep_learn_data(deep_transformer, data=vectors, target_feature_number=200, verbose=1)
# save model to file so it can be used later again

print "Saving object"
save_object(deep_transformer, r"../output/deep_learning_transformer.pkl")

