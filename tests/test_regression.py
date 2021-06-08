from banalytics.regression.linear import generate
from banalytics.regression import *
import numpy as np

class TestRegression:
    def test_generate(self):
        assert generate(2).shape[0] == 2
        assert generate(3).shape[0] == 3
        assert generate(4).shape[0] == 4

        assert generate(1337).shape[0] == 1337