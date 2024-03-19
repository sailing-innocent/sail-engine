import pytest 

import numpy as np 

@pytest.mark.current 
def test_randn():
    a = np.random.randn(5,2)
    assert len(a.shape) == 2
    assert a.shape[0] == 5
    assert a.shape[1] == 2
    