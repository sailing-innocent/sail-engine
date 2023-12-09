import pytest 
from typing import NamedTuple

class Point:
    x: int
    y: int
    def __init__(self, x, y):
        self.x = x
        self.y = y

@pytest.mark.func
def test_named_tuple():
    point = Point(1, 2)
    assert point.x == 1
    assert point.y == 2
    point.x = 3
    assert point.x == 3