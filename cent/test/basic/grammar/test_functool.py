import pytest 
import functools 

@pytest.mark.func
def test_cache():
    @functools.cache
    def factorial(n):
        return n * factorial(n-1) if n else 1
    assert factorial(10) == 3628800
    assert factorial(5) == 120
    assert factorial(12) == 479001600
