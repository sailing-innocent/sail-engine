import pytest 
from .integration import riemann_sum_np 
import numpy as np 

@pytest.mark.app 
def test_riemann_sum():
    f = lambda x: np.cos(x)
    x_start = np.pi 
    x_end = np.pi * 2 
    n = 1000
    expected = 0
    result = riemann_sum_np(f, x_start, x_end, n)
    assert np.isclose(result, expected, atol=1e-2)