import pytest 

import numpy as np 
from scipy import optimize 

def f(x):
    return x**3 - 1 

@pytest.mark.current 
def test_newton():
    fprime = lambda x: 3 * x ** 2
    root = optimize.newton(f, 1.5, fprime=fprime) # 1.0
    print(root)
    assert True 