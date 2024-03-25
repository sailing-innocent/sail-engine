import pytest 
import numpy as np 

@pytest.mark.func
def test_einsum():
    a = np.array([
        [[1,2],[3,4], [5,6]]
    ])
    assert a.shape == (1, 3, 2)
    b = np.array([[4,5],[6,7]])
    c = np.einsum('ij,klj->kli', b, a)
    assert c.shape == (1, 3, 2)
    for i in range(a.shape[0]):
        for j in range(a.shape[1]):
            s = b @ a[i][j]
            assert s[0] == c[i][j][0]
            assert s[1] == c[i][j][1]

    # print(a @ b.T) # a more convinient way
    assert True 