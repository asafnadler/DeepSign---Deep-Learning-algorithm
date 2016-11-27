import pickle

def load_transformer(fname='../output/deep_learning_transformer.pkl'):
    """
    Load the DBN transformer
    :param fname: File name of the DBN transformer
    :return: DBN transformer object
    """
    with open(fname, 'rb') as f:
        data = pickle.load(f)
        return data
