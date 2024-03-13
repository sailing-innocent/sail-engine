class A(object):
    def set(self, a, b):
        x = a
        a = b
        b = x 
        return a, b

def test_attr():
    a = A()
    c = getattr(a, 'set')
    a_='1'
    b_='2'
    _a, _b = c(a_,b_)
    assert _a == b_ 
    assert _b == a_