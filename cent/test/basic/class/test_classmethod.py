import pytest 
from math import hypot

class Vector:
    def __init__(self, x = 0, y = 0):
        self.x = x
        self.y = y

    def __repr__(self):
        return 'Vector(%r,%r)' % (self.x, self.y)

    def __abs__(self):
        return hypot(self.x, self.y)

    def __bool__(self):
        # for if(Vector)
        return bool(abs(self))
    
    def __add__(self, other):
        # for + 
        x = self.x + other.x
        y = self.y + other.y
        return Vector(x, y)

    def __mul__(self, scalar):
        # for * 
        return Vector(self.x * scalar, self.y * scalar)

    def __getitem__(self, num):
        if num == 0:
            return self.x 
        if num == 1:
            return self.y
        
@pytest.mark.func 
def test_vector():
    vec1 = Vector(1, 2)
    vec2 = Vector(3, 4)
    vec3 = vec1 + vec2
    assert vec3[0] == 4
    assert vec3[1] == 6
    assert abs(vec2) == 5

class A:
    def __init__(self, a = 1):
        self.m_a = a
    
    @staticmethod # static method is kind of seperated init function
    def a_static():
        return 1

    def a(self):
        return self.m_a

class B(A):
    @classmethod # class method is kind of init function
    def get_a(cls, a, b):
        return cls(a + b)

@pytest.mark.func
def test_static():
    assert A.a_static() == 1
    assert A().a_static() == 1
    assert A().a() == 1

    b = B.get_a(1, 2)
    assert b.a() == 3

