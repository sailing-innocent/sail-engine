import pytest 
from .model import GaussianModel

@pytest.mark.current 
def test_gaussian_model():
    gaussians = GaussianModel(3)
    assert True 