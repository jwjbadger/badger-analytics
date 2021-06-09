import numpy as np

def zscore(input):
    """
    zscore(input)

    Standardizes the data stored within `input`.

    Parameters
    ----------
    input : numpy.array
        a numpy array to standardize

    Returns
    -------
    output : numpy.array
        a numpy array with the standardized values of the original array
    """

    if input.shape[1] == 1:
        i = input.T
        return np.round((i - np.mean(i))/np.std(i), 6).T
    else:
        result = np.hsplit(input, input.shape[1]) # Split the input into it's columns
        for i in range(len(result)): # Iterate through each column
            col = result[i] # Create a variable which stores the column
            result[i] = (col - np.mean(col))/np.std(col) # Set the ith column in the result to the z score of the ith column
        return np.round(np.hstack(result), 6) # Stack the columns properly

def normalize(input):
    """
    normalize(input)

    Normalizes the data stored within `input`.

    Parameters
    ----------
    input : numpy.array
        a numpy array to normalize

    Returns
    -------
    output : numpy.array
        a numpy array with the normalized values of the original array
    """
    if input.shape[1] == 1:
        i = input.T
        return np.round((i-min(i))/(max(i)-min(i)), 6).T
    else:
        result = np.hsplit(input, input.shape[1]) # Split the input into it's columns
        for i in range(len(result)): # Iterate through each column
            col = result[i] # Create a variable which stores the column
            result[i] = (col-min(col))/(max(col)-min(col)) # Set the ith column in the result to the normalization of the ith column
        return np.round(np.hstack(result), 6) # Stack the columns properly