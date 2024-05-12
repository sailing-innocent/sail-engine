import pytest 
import torch 
import numpy as np 

@pytest.mark.func
def test_norm():
    a = torch.arange(3, dtype=torch.float) - 1
    assert torch.equal(a, torch.Tensor([-1.,0.,1.]))
    a_norm = torch.linalg.norm(a)
    a = a / a_norm
    assert torch.equal(a, torch.Tensor([-np.sqrt(2)/2, 0, np.sqrt(2)/2]))