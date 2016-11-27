from keras.models import Sequential
from keras.layers.core import Dense, Dropout #, AutoEncoder
from reduction_functions import *
from sklearn.base import TransformerMixin


# creates a keras sequential layer with given parameters
def create_sequential(input_dim, output_dim, activation, dropout,
                      w_regularizer=None, b_regularizer=None, activity_regularizer=None):
    sequential = Sequential([Dense(output_dim=output_dim, activation=activation, input_dim=input_dim,
                                   W_regularizer=w_regularizer, b_regularizer=b_regularizer,
                                   activity_regularizer=activity_regularizer)])
    if 0 < dropout < 1:
        sequential.add(Dropout(dropout))
    return sequential


# creates an encoder
def create_encoder(input_dim, output_dim, activation, dropout,
                   w_regularizer=None, b_regularizer=None, activity_regularizer=None):
    print "\tCreating Encoder..."
    encoder = create_sequential(input_dim=input_dim, output_dim=output_dim, activation=activation, dropout=dropout,
                                w_regularizer=w_regularizer, b_regularizer=b_regularizer,
                                activity_regularizer=activity_regularizer)
    print "\tEncoder created."
    return encoder


# creates an decoder
def create_decoder(input_dim, output_dim, activation, dropout,
                   w_regularizer=None, b_regularizer=None, activity_regularizer=None):
    print "\tCreating Decoder..."
    decoder = create_sequential(input_dim=input_dim, output_dim=output_dim, activation=activation, dropout=dropout,
                                w_regularizer=w_regularizer, b_regularizer=b_regularizer,
                                activity_regularizer=activity_regularizer)
    print "\tDecoder created."
    return decoder


# creates an auto-encoder from an encoder and a decoder
def create_auto_encoder(encoder, decoder):
    print "\tCreating Auto Encoder..."
    auto_encoder = AutoEncoder(encoder=encoder, decoder=decoder, output_reconstruction=True)
    print "\tAuto Encoder created."
    return auto_encoder


# compiles a giver model with a loss function and optimizer given to it (doesn't start to fit on data yet)
def compile_model(model, loss, optimizer):
    print "Compiling model..."
    model.compile(loss=loss, optimizer=optimizer)
    print "Finished compiling model."


# denoising algorithm makes the data more noisy by zeroing values from data according to given ratio
# as described in deepsign article
def denoise_data(data, ratio):
    import random
    for i in range(len(data)):
        for j in range(len(data[i])):
            if random.random() < ratio:
                data[i][j] = 0


# fit the model
def fit_model(model, x, y, denoise_ratio=0, batch_size=128, nb_epoch=100, verbose=1):
    print "Fitting model..."
    if 0 < denoise_ratio < 1:
        denoise_data(x, denoise_ratio)
    model.fit(X=x, y=y, batch_size=batch_size, nb_epoch=nb_epoch, verbose=verbose)
    print "Finished fitting model."


# After the model was trained its encoder is the only relevant part
# therefore the encoder alone should be compiled again with same parameters.
def get_trained_encoder(encoder, loss, optimizer):
    print "Creating trained encoder model..."
    trained_encoder = Sequential()
    trained_encoder.add(encoder)
    trained_encoder.compile(loss=loss, optimizer=optimizer)
    print "Trained Encoder model created."
    return trained_encoder


# Fit one iteration of the model - reduce number of feature to given amount of reduction features
# This method is called iteratively to reduce the feature count iteratively
def fit_iteration(x, reduction_count, activation='relu', loss='mse', optimizer='rmsprop', input_dropout=0,
                  hidden_dropout=0, denoise_ratio=0, batch_size=128, nb_epoch=100,
                  w_regularizer=None, b_regularizer=None, activity_regularizer=None, verbose=1):
    dimension = len(x[0])
    print "Received data with " + repr(dimension) + " features."
    print "Reducing data to " + repr(reduction_count) + " features."
    model = Sequential()
    encoder = create_encoder(input_dim=dimension, output_dim=reduction_count, activation=activation,
                             dropout=input_dropout, w_regularizer=w_regularizer, b_regularizer=b_regularizer,
                             activity_regularizer=activity_regularizer)
    decoder = create_decoder(input_dim=reduction_count, output_dim=dimension, activation=activation,
                             dropout=hidden_dropout, w_regularizer=w_regularizer, b_regularizer=b_regularizer,
                             activity_regularizer=activity_regularizer)
    model.add(create_auto_encoder(encoder=encoder, decoder=decoder))
    compile_model(model=model, loss=loss, optimizer=optimizer)
    fit_model(model=model, x=x, y=x, denoise_ratio=denoise_ratio, batch_size=batch_size, nb_epoch=nb_epoch,
              verbose=verbose)
    return get_trained_encoder(encoder=encoder, loss=loss, optimizer=optimizer)


# After a model was trained on the data, it is needed to use the trained model in order to calculate the next layer
# The function receives an encoder model and encode it to the next layer
def encode_iteration(encoder_model, x, batch_size=128):
    print "Encoding " + repr(len(x[0])) + " features..."
    result = encoder_model.predict(x, batch_size=batch_size)
    print "Encoded to " + repr(len(result[0])) + " features."
    return result


# A class of a deep transformer - extends sicit-learn TransformerMixin
class DeepTransformer(TransformerMixin):

    # Private per run parameters
    target_feature_number = None
    verbose = 1
    models = None

    # Public constant parameters - should be constants
    reduction_function = half_reduction_function
    activation = 'relu'
    loss = 'mse'
    optimizer = 'rmsprop'
    input_dropout = 0
    hidden_dropout = 0
    denoise_ratio = 0
    batch_size = 128
    nb_epoch = 100
    w_regularizer = None
    b_regularizer = None
    activity_regularizer = None

    def __init__(self):
        pass

    # Get parameters of the object
    def get_params(self):
        return {"reduction_function": self.reduction_function, "activation": self.activation, "loss": self.loss,
                "optimizer": self.optimizer, "input_dropout": self.input_dropout, "hidden_dropout": self.hidden_dropout,
                "denoise_ratio": self.denoise_ratio, "batch_size": self.batch_size, "nb_epoch": self.nb_epoch,
                "w_regularizer": self.w_regularizer, "b_regularizer": self.b_regularizer,
                "activity_regularizer": self.activity_regularizer}

    # Set parameters for the object
    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self

    # Fit the models with given data into target feature number
    def fit(self, x, target_feature_number, verbose=1):
        self.target_feature_number = target_feature_number
        self.verbose = verbose
        self.models = []
        layer = x
        while len(layer[0]) > target_feature_number:
            iteration_target_count = self.reduction_function(layer)
            model = fit_iteration(x=layer, reduction_count=iteration_target_count, activation=self.activation,
                                  loss=self.loss, optimizer=self.optimizer, input_dropout=self.input_dropout,
                                  hidden_dropout=self.hidden_dropout, denoise_ratio=self.denoise_ratio,
                                  batch_size=self.batch_size, nb_epoch=self.nb_epoch, w_regularizer=self.w_regularizer,
                                  b_regularizer=self.b_regularizer, activity_regularizer=self.activity_regularizer,
                                  verbose=self.verbose)
            self.models.append(model)
            layer = encode_iteration(model, layer, batch_size=self.batch_size)
        return self

    # Transform new data after model was trained
    def transform(self, x):
        for model in self.models:
            x = encode_iteration(model, x, batch_size=self.batch_size)
        return x

    # performs both fir and transforms data
    def fit_transform(self, X, y=None, **fit_params):
        return self.fit(X, **fit_params).transform(X)
