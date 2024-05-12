import pytest 
import torch 

@pytest.mark.env
def test_cuda_available():
    assert torch.cuda.is_available()