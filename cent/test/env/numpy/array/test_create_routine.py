import pytest
import numpy as np 


@pytest.mark.func 
def test_stack():
    a = np.zeros((2, 3))
    b = np.ones((2, 3))
    c = np.stack([a, b], axis=0)
    # stack
    assert c.shape == (2, 2, 3)
    ex_a = np.expand_dims(a, axis=0)
    assert ex_a.shape == (1, 2, 3)
    # append 
    ex_c = np.append(ex_a, c, axis=0)  
    assert ex_c.shape == (3, 2, 3)

@pytest.mark.func 
def test_repeat():
    a = np.array([1, 2, 3])
    b = np.repeat(a, 2, axis=0)
    assert b.shape == (6, ) # 1 1 2 2 3 3 
    c = np.expand_dims(a, axis=0)
    assert c.shape == (1, 3)
    d = np.repeat(c, 2, axis=0)
    assert d.shape == (2, 3) # [[1,2,3],[1,2,3]]


# linspace
@pytest.mark.func
def test_linspace():
    a = np.linspace(0, 10, 5)
    assert np.all(a == [0, 2.5, 5, 7.5, 10])

# meshgrid
@pytest.mark.func 
def test_meshgrid():
    i, j = np.meshgrid(np.arange(3), np.arange(2))
    assert i.shape == (2, 3)
    assert j.shape == (2, 3)

    print(i)
    print(j)


# arange
@pytest.mark.func
def test_nparange():
    lista = np.arange(3)
    assert lista.shape == (3,)
    assert lista[0] == 0
    assert lista[1] == 1
    assert lista[2] == 2

# random choise

@pytest.mark.func 
def test_np_random_choise():
    lista = np.random.choice(3, 2)
    print(lista)
    # assert lista.shape == (2,)

    N = 10
    M = 3
    batch_size = 8
    p = np.zeros(N)
    start = (N-M)//2
    for i in range(start, start+M):
        p[i] = 1/M
    assert p.shape == (N,)
    assert p[0] == 0
    assert p[start] == 1/M 
    assert p[start+M-1] == 1/M 
    print(p)
    lista = np.random.choice(N, batch_size, p=p)
    print(lista)
    assert lista.shape == (batch_size,)