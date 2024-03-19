import pytest 
import numpy as np

@pytest.mark.func
def test_ravel():
    a = np.arange(12).reshape(3, 4)
    assert a.shape == (3, 4)
    b = a.ravel()
    assert b.shape == (12,)