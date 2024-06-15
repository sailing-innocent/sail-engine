import pytest 
from sailcupy import add

@pytest.mark.app
def test_ing_add():
    assert add(1, 2) == 3