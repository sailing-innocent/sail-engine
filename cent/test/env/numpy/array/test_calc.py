import pytest
import numpy as np 

@pytest.mark.func 
def test_arr_dot():
    M = np.array([[1,2],[3,4]])
    x = np.array([[5,6],[7,8],[9,10]])
    z = np.dot(x, M)
    print(z)
    assert True 

@pytest.mark.func 
def test_np_arr_mul():
    a = np.array([[1,2],[3,4]])
    b = np.array([[5,6],[7,8]])
    c = a*b

    assert c.shape == (2,2)
    assert c[0,0] == 5
    assert c[0,1] == 12
    assert c[1,0] == 21
    assert c[1,1] == 32

@pytest.mark.func 
def test_mat_prod():
    mat = np.array([[1, 1], [0, 1]])
    p = np.array([2, 3])
    # { 1  1 } @ {  2  } = {  5  } 
    #   0  1        3         3
    assert np.all(mat @ p == np.array([5, 3])) # auto treat p as column vector
    assert np.all(p @ mat == np.array([2, 5])) # auto treat p as row vector

@pytest.mark.func 
def test_mm():
    m1 = np.array([[1,2],[3,4]])
    m2 = np.array([[5,6],[7,8]])
    m3 = m1 @ m2 
    m4 = m2 @ m1 
    assert np.all(m3 == np.array([[19, 22], [43, 50]]))
    assert np.all(m4 == np.array([[23, 34], [31, 46]]))