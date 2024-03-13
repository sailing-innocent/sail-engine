import pytest 
from abc import ABC, abstractmethod 

class Base(ABC):
    def __init__(self):
        self.hello = "hello Base"

    @abstractmethod
    def get_count(self):
        return 1

class DerivedA(Base):
    def __init__(self):
        super(DerivedA, self).__init__()

    def get_count(self):
        return 2

class DerivedB(Base):
    def __init__(self):
        super(DerivedB, self).__init__()

    def get_count(self):
        return 3


def call_base(b: Base):
    return b.get_count()

@pytest.mark.func 
def test_abc():
    da = DerivedA()
    db = DerivedB()
    assert da.get_count() == 2
    assert call_base(da) == 2
    assert db.get_count() == 3
    assert call_base(db) == 3

