import pytest 
import torch 

@pytest.mark.app
def test_prod():
    m1 = torch.tensor([[1,2],[3,4]])
    m2 = torch.tensor([[5,6],[7,8]])
    m3 = m1 @ m2
    m4 = m2 @ m1
    assert torch.all(m3 == torch.tensor([[19, 22], [43, 50]]))
    assert torch.all(m4 == torch.tensor([[23, 34], [31, 46]]))
    v = torch.tensor([1, 2])
    v1 = v @ m1
    v2 = m1 @ v
    assert torch.all(v1 == torch.tensor([7, 10]))
    assert torch.all(v2 == torch.tensor([5, 11]))