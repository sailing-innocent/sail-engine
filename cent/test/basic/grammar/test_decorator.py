import pytest 

def a_new_dec(a_func, *args, **kwargs):
    def wrapper(*args, **kwargs):
        return a_func(1, *args, **kwargs)
    return wrapper

@a_new_dec
def inc_return(a, b):
    return a + b + 1

@pytest.mark.func  
def test_dec():
    res = inc_return(2)
    assert res == 4 # 1 + 2 + 1 = 4