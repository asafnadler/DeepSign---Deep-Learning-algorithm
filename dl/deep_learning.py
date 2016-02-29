# This file contains functions to integrate the reading of a csv file and performing deep learning on it.
from csv_handler import *


# the function receives a plain data and a DeepTransformer and applies it on the data.
# Parameters for fitting - reduction size and verbose.
def deep_learn_data(deep_transformer, data, **fit_params):
    return deep_transformer.fit_transform(data, **fit_params)


# Receives a csv file path and a DeepTransformer object and some parameters:
# Remove columns - which columns to remove from the deep-leaning mechanism.
# Ids column if exists.
# Responses column if exists.
# Parameters for fitting - reduction size and verbose.
# The function reads the CSV perform the deep learning algorithm on it, and write it back to result file path.
def deep_learn_csv(csv_path, deep_transformer, result_file_path=None, remove_columns=None, ids=None, responses=None,
                   **fit_params):
    data = read_csv(csv_path)
    if not remove_columns:
        remove_columns = []
    removed_columns = []
    for column in remove_columns:
        removed_columns.append([column, data[column]])
        data = data.drop(column, axis=1)
    ids_column = None
    if ids:
        ids_column = data[ids]
        data = data.drop(ids, axis=1)
    responses_column = None
    if responses:
        responses_column = data[responses]
        data = data.drop(responses, axis=1)
    result = deep_learn_data(deep_transformer, data.values, **fit_params)
    if result_file_path:
        write_to_csv(result=result, result_file_path=result_file_path, removed_columns=removed_columns,
                     responses=responses, responses_column=responses_column, ids_column=ids_column)

