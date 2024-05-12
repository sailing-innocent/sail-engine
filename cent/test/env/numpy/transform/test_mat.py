import pytest 

import numpy as np 

@pytest.mark.app
def test_mt():
    # numpy is row-major
    # m = 1 2 
    #     3 4
    m = np.array([[1, 2], [3, 4]])
    assert m[0][0] == 1
    assert m[0][1] == 2
    assert m[1][0] == 3
    assert m[1][1] == 4
    m_flatten = m.flatten().tolist()
    assert m_flatten[0] == 1
    assert m_flatten[1] == 2
    assert m_flatten[2] == 3
    assert m_flatten[3] == 4

    mt = m.T
    assert mt[0][0] == 1
    assert mt[0][1] == 3
    assert mt[1][0] == 2
    assert mt[1][1] == 4

@pytest.mark.app
def test_mv():
    # matrix vector multiplication
    # numpy is row-major
    # m = 1 2 x 1 = 5
    #     3 4   2   11
    m = np.array([[1, 2], [3, 4]])
    v = np.array([1, 2])
    mv = np.matmul(m, v)
    assert mv[0] == 5
    assert mv[1] == 11

@pytest.mark.current 
def test_mm():
    # matrix-matrix multiplication
    # numpy is row-major
    # m = 1 2 x 1 2 = 7 10
    #     3 4   3 4   15 22

    m = np.array([[1, 2], [3, 4]])
    m2 = np.array([[1, 2], [3, 4]])
    mm = np.matmul(m, m2)
    assert mm[0][0] == 7
    assert mm[0][1] == 10
    assert mm[1][0] == 15
    assert mm[1][1] == 22
