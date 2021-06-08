import numpy as np

def standardization(input_set):
    """
    standardization(input_set)

    Standardizes the data stored within `input_set`.

    Parameters
    ----------
    input_set : numpy.array
        a numpy array to standardize

    Returns
    -------
    output : numpy.array
        a numpy array with the standardized values of the original array
    """

    result = np.array([])
    X = np.hsplit(input_set, len(input_set[0])) # split the input array into columns
    for i in X:
        result = np.append(result, ((i - np.mean(i))/np.std(i)))
    return result

def normalization(input_set):
    """
    normalization(input_set)

    Normalizes the data stored within `input_set`.

    Parameters
    ----------
    input_set : numpy.array
        a numpy array to normalize

    Returns
    -------
    output : numpy.array
        a numpy array with the normalized values of the original array
    """

    result = np.array([])
    X = np.hsplit(input_set, len(input_set[0])) # split the input array into columns
    for i in X:
        result = np.append(result, (i-min(i))/(max(i)-min(i)))
    return result