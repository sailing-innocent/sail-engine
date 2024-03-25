# Test Numpy Math Operation
import numpy as np 
import pytest 

@pytest.mark.func 
def test_frac():
    a = 1.2
    a_fl = np.floor(a)
    a_fc = a - a_fl
    assert np.abs(a_fc - 0.2) < 0.0001
    assert a_fl == 1.0

@pytest.mark.func 
def test_at_operator():
    # A @ B is the matrix multiplication
    a = np.array([[1,2],[3,4]])
    b = np.array([[4,5],[6,7]])
    c = a @ b 
    assert c[0][0] == 16
    assert c[0][1] == 19
    assert c[1][0] == 36
    assert c[1][1] == 43

@pytest.mark.func 
def test_dot():
    # np.dot(A,B) is the matrix multiplication
    a = np.array([[1,2],[3,4]])
    b = np.array([[4,5],[6,7]])
    c = np.dot(a,b) 
    assert c[0][0] == 16
    assert c[0][1] == 19
    assert c[1][0] == 36
    assert c[1][1] == 43

@pytest.mark.func 
def test_diff():
    a = np.array([1,2,3,6,5])
    b = np.diff(a)
    assert b.size == 4
    assert b[0] == 1
    assert b[1] == 1
    assert b[2] == 3
    assert b[3] == -1

@pytest.mark.func
def test_tile():
    a = np.array([[1,2],[3,4]])
    b = np.tile(a, (2, 2))
    assert b.shape == (4, 4)
    print(b) # [3,4,3,4][1,2,1,2][3,4,3,4]

@pytest.mark.to
def test_broadcast_to():
    assert True 