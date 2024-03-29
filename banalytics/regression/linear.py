import numpy as np

class LinearModel:
    """
    __init__(self, x, y, theta=None)

    A class for linear regression models, Takes in an x matrix and y vector to use for a variety of regression functions.

    Parameters
    ----------
    x : numpy.array
        the x matrix
    y : numpy.array
        the y matrix
    theta : numpy.array, optional
        the theta vector
    """

    def __init__(self, x, y, theta=None):
        self.y = y

        if type(theta) == np.ndarray:
            self.theta = theta
        else:
            self.generate(x.shape[1]+1) # Creates theta only if theta isn't a parameter
        self.m = x.shape[0] # number of rows in x
        self.X = np.hstack((np.ones((self.m,1)), x)) # x with a prepended column of ones

    def generate(self, n):
        """
        generate(n)

        Generates a random theta vector of size n with a uniform distribution over [0, 10).

        Parameters
        ----------
        n : int
            an integer that describes the size of the resulting theta vector

        Returns
        -------
        self
        """

        self.theta = np.multiply(np.random.rand(n, 1), 10)
        return self

    def squared_error(self, theta=None):
        """
        squared_error(self, theta=None)

        Gives the squared error for the given x matrix, y vector and theta vector.

        Parameters
        ----------
        theta : numpy.array, optional
            optional theta vector to use instead of `self.theta`

        Returns
        -------
        output : float
            a float of the squared error with the given parameters
        """

        if type(theta) != np.ndarray:
            theta = self.theta
        h = np.dot(self.X, theta) # h(x)

        return np.round((1/(2*self.m)) * np.sum(np.power((np.subtract(h, self.y)), 2)), 6)

    def gradient_descent(self, it, alpha, n=None): # squared error only currently
        """
        gradient_descent(self, it, alpha, n=None):

        Runs gradient descent on a specific x matrix and y vector with i iterations and a learning rate, alpha.

        Parameters
        ----------
        it : int
            the number of times for gradient descent to run
        alpha : float
            The learning rate for gradient descent should be chosen such that it isn't too small or too large. If it is too small, gradient descent will be slow. If it is too large, gradient descent may overshoot and end with a bad error rate.
        n : int, optional
            an optional parameter for how many times to run; returns the best theta vector

        Returns
        -------
        self
        """

        if n: # if n exists, we should run a loop
            best_theta = self.theta

            for j in range(n):
                self.generate(self.X.shape[1]) # regenerate theta
                temp = np.ndarray((self.X.shape[1], 1))

                for k in range(1, it):
                    h = np.dot(self.X, self.theta) # h(x)

                    for i in range(temp.shape[1] + 1):
                        temp[i][0] = self.theta[i][0] - (alpha * (1/self.m) * np.sum(np.multiply(np.subtract(h, self.y), self.X[:,[i]])))
                    self.theta = temp
                
                # if this theta is better than the best, make the best this theta
                if self.squared_error() < self.squared_error(theta=best_theta):
                    best_theta = self.theta

            self.theta = np.round(best_theta, 6)
            return self

        temp = np.ndarray((self.X.shape[1], 1))

        for j in range(1, it):
            h = np.dot(self.X, self.theta) # h(x)

            for i in range(temp.shape[1] + 1):
                temp[i][0] = self.theta[i][0] - (alpha * (1/self.m) * np.sum(np.multiply(np.subtract(h, self.y), self.X[:,[i]])))
            self.theta = temp
        
        self.theta = np.round(self.theta, 6)
        return self
    
    def normal(self):
        """
        normal(self)

        Runs a normal equation on X and y to create a theta matrix. This is extremely consistent and willl give good results within good times for roughly m < 10000 (m is the number of training examples in X).

        Returns
        -------
        self
        """

        self.theta = np.round(np.dot(np.dot(np.linalg.pinv(np.dot(self.X.T, self.X)), self.X.T), self.y), 6)
        return self