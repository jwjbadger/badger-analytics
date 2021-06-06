import numpy as np

# generates theta matrix 
def generate(n):
    return np.multiply(np.random.rand(n), 10)

def squared_error(x, y, theta):
    m = x.shape[0] # number of rows in x
    X = np.hstack((np.ones((m,1)),x)) # x with a prepended column of ones
    h = np.dot(X, theta) # h(x)

    return (1/(2*m)) * np.sum(np.power((np.subtract(h, y)), 2))