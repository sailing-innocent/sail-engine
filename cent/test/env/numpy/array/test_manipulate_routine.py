import pytest
import numpy as np 

@pytest.mark.func 
def test_r_c_():
    a = np.array([[1,2],[3,4]])
    b = np.array([[5,6],[7,8]])
    c = np.r_[a, b]
    c_expected = np.array([[1,2],[3,4],[5,6],[7,8]])
    assert np.all(c == c_expected)

    d = np.c_[a,b]
    d_expected = np.array([[1,2,5,6],[3,4,7,8]])
    assert np.all(d == d_expected)