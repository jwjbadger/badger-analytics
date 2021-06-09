from banalytics.regression import PolynomialModel
import numpy as np

class TestPolynomialModel:
    def test_generate(self):
        # Testing smaller, common numbers result in the right size
        assert PolynomialModel(np.ndarray((4, 2)), np.ndarray((4, 1))).theta.shape[0] == 3
        assert PolynomialModel(np.ndarray((4, 3)), np.ndarray((4, 1))).theta.shape[0] == 4
        assert PolynomialModel(np.ndarray((4, 4)), np.ndarray((4, 1))).theta.shape[0] == 5

        # Testing larger numbers result in the right size
        assert PolynomialModel(np.ndarray((4, 1336)), np.ndarray((4, 1))).theta.shape[0] == 1337 
    
    def test_squared_error(self):
        x = np.array([[1], [2], [3], [4], [5]])
        y = np.array([[6], [15], [28], [45], [66]])
        theta = np.array([[1], [3], [2]])
        assert PolynomialModel(x, y, theta=theta).squared_error() == 0 # Squared error of equal functions and lists should be 0 (squared error should work for matrices)

        x = np.array([[1], [2], [3], [4], [5]])
        y = np.array([[6], [16], [24], [44], [66]])
        theta = np.array([[1], [3], [2]])
        assert PolynomialModel(x, y, theta=theta).squared_error() == 1.8 # Using previously calculated squared error (by hand)

        x = np.array([[2, 1], [4, 2], [6, 3], [8, 4], [10, 5]])
        y = np.array([[6], [15], [28], [45], [66]])
        theta = np.array([[1], [1.5], [2]])
        assert PolynomialModel(x, y, theta=theta).squared_error() == 0 # Using previously calculated squared error (by hand) (squared error should work for matrices)
    
    # The following test may take a long time, remove it if testing other items
    def test_gradient_descent(self):
        x = np.array([[1], [2], [3], [4], [5]])
        y = np.array([[5], [11], [19], [29], [41]])
        model = PolynomialModel(x, y)
        model.gradient_descent(10000, 0.06, n=15)
        assert model.squared_error() < 0.1 # because polynomial regression is more complex, it is much less likely to converge properly

        x = np.array([[2, 1], [4, 2], [6, 3], [8, 4], [10, 5]])
        y = np.array([[5], [11], [19], [29], [41]])
        model = PolynomialModel(x, y)
        model.gradient_descent(10000, 0.03, n=15)
        assert model.squared_error() < 0.1