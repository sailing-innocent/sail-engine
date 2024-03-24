import pytest 

from .classify import ClassifyAccuracy

@pytest.mark.func
def test_dummy_classify():
    target = [0, 1, 2, 1, 2, 1]
    predicted = [1, 1, 0, 1, 1, 1]
    N = len(target)
    accuracy = ClassifyAccuracy(target, predicted, N)
    assert accuracy.value == 0.5
    assert str(accuracy) == "accuracy: 0.5"