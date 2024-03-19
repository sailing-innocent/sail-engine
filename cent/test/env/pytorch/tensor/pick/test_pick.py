import pytest 
import torch 

@pytest.mark.current 
def test_pick():
    A = torch.arange(24).reshape(4, 2, 3)
    B = A[:, 1, :]
    assert torch.equal(B, torch.tensor([[3, 4, 5], [9, 10, 11], [15, 16, 17], [21, 22, 23]]))
