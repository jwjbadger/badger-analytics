from banalytics.regression.linear import generate, squared_error
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