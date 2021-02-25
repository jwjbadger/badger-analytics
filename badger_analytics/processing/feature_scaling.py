import numpy as np

def standardization(input_set):
    result = np.array([])
    X = np.hsplit(input_set, len(input_set[0]))
    for i in X:
        result = np.append(result, ((i - np.mean(i))/np.std(i)))
    return result

def normalization(input_set):
    return