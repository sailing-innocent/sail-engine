import pytest 
import numpy as np 

@pytest.mark.current 
def test_mm():
    # 1, 2
    # 3, 4
    m1 = np.array([[1,2],[3,4]])
    # 5, 6
    # 7, 8
    m2 = np.array([[5,6],[7,8]])
    # 1 2 x 5 6 = 19 22
    # 3 4   7 8   43 50
    m3 = m1 @ m2 
    # 5 6 x 1 2 = 23 34
    # 7 8   3 4   31 46
    m4 = m2 @ m1 
    assert np.all(m3 == np.array([[19, 22], [43, 50]]))
    assert np.all(m4 == np.array([[23, 34], [31, 46]]))