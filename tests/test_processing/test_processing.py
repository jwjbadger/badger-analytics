import numpy as np
from banalytics.processing.feature_scaling import *

class TestProcessing:
    def test_zscore(self):
        # Z score should work for vectors
        assert np.array_equal(zscore(np.array([[48], [52], [57], [78]])), np.array([[-0.929743], [-0.583792], [-0.151354], [1.664889]])) == True 
        assert np.array_equal(zscore(np.array([[30], [600], [350], [824]])), np.array([[-1.425743], [0.504598], [-0.342043], [1.263188]])) == True

        # Z score should also work for matrices
        assert np.array_equal(zscore(np.array([[48, 48], [52, 52], [57, 57], [78, 79]])), np.array([[-0.929743, -0.918262], [-0.583792, -0.584349], [-0.151354, -0.166957], [1.664889, 1.669568]])) == True
        assert np.array_equal(zscore(np.array([[23, 980], [670, 456], [736, 536], [1, 87]])), np.array([[-0.965718, 1.465162], [0.902203, -0.185015], [1.092748, 0.06692], [-1.029233, -1.347067]])) == True

    def test_normalization(self):
        # Min max normalization should work for vectors
        assert np.array_equal(normalize(np.array([[48], [52], [57], [78]])), np.array([[0], [0.133333], [0.3], [1]])) == True 
        assert np.array_equal(normalize(np.array([[30], [600], [350], [824]])), np.array([[0], [0.717884], [0.403023], [1]])) == True

        # Normalization should also work for matrices
        assert np.array_equal(normalize(np.array([[48, 48], [52, 52], [57, 57], [78, 79]])), np.array([[0, 0], [0.133333, 0.129032], [0.3, 0.290323], [1, 1]])) == True
        assert np.array_equal(normalize(np.array([[23, 980], [670, 456], [736, 536], [1, 87]])), np.array([[0.029932, 1], [0.910204, 0.413214], [1, 0.5028], [0, 0]])) == True

