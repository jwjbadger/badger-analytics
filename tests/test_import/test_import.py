from banalytics.data_import import *
import numpy as np

def test_import():
    # remote import not tested for simplicity's sake (has been tested locally)
    assert np.array_equal(local_import("tests/test_import/ex_import"), np.array([[1, 2, 3], [4, 5, 6]])) == True # local import should work when bracekts are seperated with newline
    assert np.array_equal(local_import("tests/test_import/ex_import_2"), np.array([[1, 2, 3], [4, 5, 6]])) == True # local import should work when brackets are normal    