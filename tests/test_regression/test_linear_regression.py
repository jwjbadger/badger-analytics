from banalytics.regression.linear import *
import numpy as np

class TestLinearRegression:
    def test_generate(self):
        # Testing smaller, common numbers result in the right size
        assert generate(2).shape[0] == 2 
        assert generate(3).shape[0] == 3
        assert generate(4).shape[0] == 4

        # Testing larger numbers result in the right size
        assert generate(1337).shape[0] == 1337 
    
    def test_squared_error(self):
        x = np.array([[1], [2], [3], [4], [5]])
        y = np.array([[2], [4], [6], [8], [10]])
        theta = np.array([[0], [2]])
        assert squared_error(x, y, theta) == 0 # Squared error of equal functions and lists should be 0

        x = np.array([[1], [2], [3], [4], [5]])
        y = np.array([[2], [5], [6], [4], [10]])
        theta = np.array([[0], [2]])
        assert squared_error(x, y, theta) == 1.7 # Using previously calculated squared error (by hand)

        x = np.array([[1, 2], [2, 4], [3, 6], [4, 8], [5, 10]])
        y = np.array([[4], [8], [12], [16], [20]])
        theta = np.array([[0], [2], [1]])
        assert squared_error(x, y, theta) == 0 # Squared error of equal functions and lists should be 0 (squared error should work for matrices)

        x = np.array([[3, 2], [2, 4], [3, 4], [4, 9], [5, 10]])
        y = np.array([[4], [8], [12], [16], [20]])
        theta = np.array([[0], [2], [1]])
        assert squared_error(x, y, theta) == 2.1 # Using previously calculated squared error (by hand) (squared error should work for matrices)
    
    def test_gradient_descent(self):
        x = np.array([[1], [2], [3], [4], [5]])
        y = np.array([[2], [4], [6], [8], [10]])
        output = gradient_descent(10000, 0.1, x, y) # for data sets with an exact line, the output should be that line given a proper amount of iterations and a well-chosen alpha
        assert output[0][0] == 0 # check that the y-intercept is 0
        assert output[1][0] == 2 # check that the slope is 2

        x = np.array([[1], [2], [3], [4], [5]])
        y = np.array([[2], [4], [5], [8], [9]])
        output = gradient_descent(10000, 0.1, x, y) # data sets without an exact line should be within a certain tolerance (in this case, the result should always be [[0.2], [1.8]])
        assert output[0][0] == 0.2 # check that the y-intercept is 0.2
        assert output[1][0] == 1.8 # check that the slope is 1.8