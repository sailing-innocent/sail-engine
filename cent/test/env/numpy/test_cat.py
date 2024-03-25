import pytest 
import numpy as np 

@pytest.mark.func
def test_stack():
    a = np.array([i for i in range(12)]).reshape(3,4)
    assert a.all() == np.array([[0,1,2,3],[4,5,6,7],[8,9,10,11]]).all()
    b = np.array([i for i in range(12,24)]).reshape(3,4)
    c = np.array([i for i in range(24,36)]).reshape(3,4)

    new_array = np.stack((a,b,c), axis=0)
    print(new_array)


@pytest.mark.func
def test_concatenate():
    a = [0,4,5,1]
    b = [3,2,6,7]
    c = np.concatenate((a,b))
    assert not np.any(c - np.array([0,4,5,1,3,2,6,7]))
    c.sort(kind="mergesort")
    assert not np.any(c - np.array([0,1,2,3,4,5,6,7]))

@pytest.mark.func 
def test_concatenate_2d():
    # np.concatenate((a,b), axis=0) will join a sequence of arrays along an existing axis
    a = np.array([[1,2],[3,4]])
    b = np.array([[4,5],[6,7]])
    c = np.concatenate((a,b), axis=0)
    assert c[0][0] == 1
    assert c[0][1] == 2
    assert c[1][0] == 3
    assert c[1][1] == 4
    assert c[2][0] == 4
    assert c[2][1] == 5
    assert c[3][0] == 6
    assert c[3][1] == 7
    d = np.concatenate((a,b), axis=1)
    assert d[0][0] == 1
    assert d[0][1] == 2
    assert d[0][2] == 4
    assert d[0][3] == 5
    assert d[1][0] == 3
    assert d[1][1] == 4
    assert d[1][2] == 6
    assert d[1][3] == 7