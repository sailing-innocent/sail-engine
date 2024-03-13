import pytest 

from dataclasses import dataclass

@dataclass(order=True) 
class Dat:
    x: int
    y: int
    def __add__(self, other):
        return Dat(self.x + other.x, self.y + other.y)

@pytest.mark.current 
def test_dataclass():
    a = Dat(1, 2)
    assert a.x == 1
    assert a.y == 2

    b = Dat(3, 4)
    assert a < b
    c = a + b
    assert c.x == 4
    assert c.y == 6