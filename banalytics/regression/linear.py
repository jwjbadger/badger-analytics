import numpy as np

# generates theta vector 
def generate(n):
    """
    generate(n)

    Generates a random theta vector of size n with a uniform distribution over [0, 10).

    Parameters
    ----------
    n : int
        an integer that describes the size of the resulting theta vector

    Returns
    -------
    output : numpy.array
        a 2d numpy array of the randomly generated theta matrix
    """

    return np.multiply(np.random.rand(n, 1), 10)

def squared_error(x, y, theta):
    """
    squared_error(x, y, theta)

    Gives the squared error for given x and y vectors and a theta vector.

    Parameters
    ----------
    x : numpy.array
        a 2d numpy array that holds the x values as a vector
    y : numpy.array
        a 2d numpy array that holds the y values as a vector
    theta : numpy.array
        a 2d numpy array that holds the theta values as a vector

    Returns
    -------
    output : float
        a float of the squared error with the given parameters
    """

    m = x.shape[0] # number of rows in x
    X = np.hstack((np.ones((m,1)),x)) # x with a prepended column of ones
    h = np.dot(X, theta) # h(x)

    return np.round((1/(2*m)) * np.sum(np.power((np.subtract(h, y)), 2)), 6)

def gradient_descent(i, alpha, x, y): # squared error only currently
    """
    gradient_descent(i, alpha, x, y, init_theta)

    Runs gradient descent on a specific x and y vector with i iterations and a learning rate, alpha.

    Parameters
    ----------
    i : int
        the number of times for gradient descent to run
    alpha : float
        The learning rate for gradient descent should be chosen such that it isn't too small or too large. If it is too small, gradient descent will be slow. If it is too large, gradient descent may overshoot and end with a bad error rate.
    x : numpy.array
        a 2d numpy array which describes the feature set
    y : numpy.array
        a 2d numpy array which holds the y values

    Returns
    -------
    output : numpy.array
        a 2d numpy array with the theta values optimized for the given x and y vectors
    """

    theta = generate(2) # set an initial theta vector
    m = x.shape[0] # number of rows in x
    X = np.hstack((np.ones((m,1)),x)) # x with a prepended column of ones

    for i in range(1, i):
        h = np.dot(X, theta) # h(x)

        temp0 = theta[0][0] - (alpha * (1/m) * np.sum(np.subtract(h, y)))
        temp1 = theta[1][0] - (alpha * (1/m) * np.sum(np.multiply(np.subtract(h, y), x)))
        theta = np.array([[temp0], [temp1]])

    return np.round(theta, 6)