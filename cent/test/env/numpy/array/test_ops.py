import pytest 
import numpy as np 

@pytest.mark.current 
def test_filter():
    a = np.array([[0,1], [2,3], [4,5]])
    b = np.array([0, 1, 1])
    c = a[b==1,0]
    c_expected = [2,4]
    assert np.all(c == c_expected)


@pytest.mark.func
def test_np_all():
    a = np.array([1,2,3,4])
    b = np.array([1,2,3,4])
    assert np.all(a==b)
    c = np.array([[1,2],[3,4]])
    d = np.array([[1,2],[3,4]])
    assert np.all(c==d)