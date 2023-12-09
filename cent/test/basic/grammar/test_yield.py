import pytest 

@pytest.mark.func
def test_yield():
    def gen():
        for i in range(10):
            yield i

    g = gen()
    for i, ig in enumerate(g):
        assert i == ig