import pytest 
import itertools

@pytest.mark.func
def test_product():
    A = [1, 2]
    B = [3, 4]
    idx = 0
    for i in itertools.product(A, B):
        idx = idx + 1
    assert idx == 4
    assert True 

@pytest.mark.func
def test_product_self():
    A = ['int32_t', 'uint32_t', 'int64_t', 'uint64_t', 'float', 'double']
    S = itertools.product(A,A)
    Sc = list(S)
    for k,v in S:
        pass
    for k,v in Sc:
        pass 
    assert len(Sc) == 36
    assert True 