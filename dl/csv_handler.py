# This file handles actions performed relevant to csv (reading and writing)

import pandas as pd
import numpy as np


# Read a csv file with given path and return a data-set which is contained there
def read_csv(file_path):
    print "Start reading CSV file"
    data = pd.read_csv(file_path)
    print "Finished reading CSV file"
    return data


# Write to csv the result values to result_file_path
# If columns have been removed from the original data set it is possible to had them back by providing "removed_columns"
# Sam for response column (provide name and data) and for id column (provide ids only)
def write_to_csv(result, result_file_path, removed_columns=[],
                 responses=None, responses_column=None, ids_column=None):
    print "Writing result to " + result_file_path
    new_columns = ["Column_" + repr(i) for i in range(len(result[0]))]
    for removedColumn in removed_columns:
        result = np.c_[result, removedColumn[1]]
        new_columns.append(removedColumn[0])
    if responses:
        result = np.c_[result, responses_column]
        new_columns.append(responses)
    result = pd.DataFrame(result, index=ids_column, columns=new_columns)
    if ids_column:
        result.to_csv(result_file_path)
    else:
        result.to_csv(result_file_path, index=False)
    return result


# Sometimes it is good to change binary values into -1/1 instead of 0/1
# Therefore this function receives a csv file path and changes all zeros to minus 1 and save it to result file path
def change_0_to_minus_1(csv_file, result_file):
    data = read_csv(csv_file).values
    for i in range(len(data)):
        for j in range(len(data[i])):
            if data[i][j] == 0:
                data[i][j] = -1
    write_to_csv(data, result_file)

