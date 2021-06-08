import numpy as np

# generates theta matrix 
def generate(n):
    return np.multiply(np.random.rand(n, 1), 10)

def squared_error(x, y, theta):
    m = x.shape[0] # number of rows in x
    X = np.hstack((np.ones((m,1)),x)) # x with a prepended column of ones
    h = np.dot(X, theta) # h(x)

    return np.round((1/(2*m)) * np.sum(np.power((np.subtract(h, y)), 2)), 6)

def gradient_descent(i, alpha, x, y, init_theta): # squared error only currently
    theta = init_theta # set a mutatable theta
    m = x.shape[0] # number of rows in x
    X = np.hstack((np.ones((m,1)),x)) # x with a prepended column of ones

    for i in range(1, i):
        h = np.dot(X, theta) # h(x)

        temp0 = theta[0][0] - (alpha * (1/m) * np.sum(np.subtract(h, y)))
        temp1 = theta[1][0] - (alpha * (1/m) * np.sum(np.multiply(np.subtract(h, y), x)))
        theta = np.array([[temp0], [temp1]])

    return theta