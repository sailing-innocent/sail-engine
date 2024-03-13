import pytest 

@pytest.mark.func 
def test_list():
    a = [0, 1]
    a.append(2)
    assert a == [0, 1, 2]

@pytest.mark.func  
def test_list_iter():
    a = [0, 1]
    b = [[2,3], [4,5]]
    c = [i for ib in b for i in ib]
    assert c == [2, 3, 4, 5]