from banalytics.regression.linear import *
import numpy as np

class TestLinearModel:
    def test_generate(self):
        # Testing smaller, common numbers result in the right size
        assert LinearModel(np.ndarray((4, 1)), np.ndarray((4, 1))).theta.shape[0] == 2 
        assert LinearModel(np.ndarray((4, 2)), np.ndarray((4, 1))).theta.shape[0] == 3
        assert LinearModel(np.ndarray((4, 3)), np.ndarray((4, 1))).theta.shape[0] == 4

        # Testing larger numbers result in the right size
        assert LinearModel(np.ndarray((4, 1336)), np.ndarray((4, 1))).theta.shape[0] == 1337 
    
    def test_squared_error(self):
        x = np.array([[1], [2], [3], [4], [5]])
        y = np.array([[2], [4], [6], [8], [10]])
        theta = np.array([[0], [2]])
        assert LinearModel(x, y, theta=theta).squared_error() == 0 # Squared error of equal functions and lists should be 0

        x = np.array([[1], [2], [3], [4], [5]])
        y = np.array([[2], [5], [6], [4], [10]])
        theta = np.array([[0], [2]])
        assert LinearModel(x, y, theta=theta).squared_error() == 1.7 # Using previously calculated squared error (by hand)

        x = np.array([[1, 2], [2, 4], [3, 6], [4, 8], [5, 10]])
        y = np.array([[4], [8], [12], [16], [20]])
        theta = np.array([[0], [2], [1]])
        assert LinearModel(x, y, theta=theta).squared_error() == 0 # Squared error of equal functions and lists should be 0 (squared error should work for matrices)

        x = np.array([[3, 2], [2, 4], [3, 4], [4, 9], [5, 10]])
        y = np.array([[4], [8], [12], [16], [20]])
        theta = np.array([[0], [2], [1]])
        assert LinearModel(x, y, theta=theta).squared_error() == 2.1 # Using previously calculated squared error (by hand) (squared error should work for matrices)
    
    def test_gradient_descent(self):
        x = np.array([[1], [2], [3], [4], [5]])
        y = np.array([[2], [4], [6], [8], [10]])
        output = LinearModel(x, y).gradient_descent(10000, 0.1) # for data sets with an exact line, the output should be that line given a proper amount of iterations and a well-chosen alpha
        assert output[0][0] == 0 # check that the y-intercept is 0
        assert output[1][0] == 2 # check that the slope is 2

        x = np.array([[1], [2], [3], [4], [5]])
        y = np.array([[2], [4], [5], [8], [9]])
        output = LinearModel(x, y).gradient_descent(10000, 0.1) # data sets without an exact line should be within a certain tolerance (in this case, the result should always be [[0.2], [1.8]])
        assert output[0][0] == 0.2 # check that the y-intercept is 0.2
        assert output[1][0] == 1.8 # check that the slope is 1.8

        x = np.array([[1, 2], [2, 4], [3, 6], [4, 8], [5, 10]])
        y = np.array([[4], [8], [12], [16], [20]])
        model = LinearModel(x, y)
        model.gradient_descent(10000, 0.1)
        assert model.squared_error() == 0 # adding another variable means there are multiple solutions, so we can't check that the solutions are the same, but we must check the squared error is 0 (because the program should come up with an exact solution)

        # The following test may take a long time, remove it if testing other items
        x = np.array([[3, 2], [2, 4], [3, 4], [4, 9], [5, 10]])
        y = np.array([[4], [8], [12], [16], [20]])
        model = LinearModel(x, y)
        model.gradient_descent(10000, 0.14, n=15) # We must loop through because this case is much less consistent to verify that the error is on average relatively good
        assert model.squared_error() < 2.1 