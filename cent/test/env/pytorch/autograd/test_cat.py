import pytest 
import torch 

@pytest.mark.current 
def test_cat():
    a = torch.Tensor([[1,2,3],[4,5,6]])
    a.requires_grad_()
    b = torch.Tensor([[7,8,9],[10,11,12]])
    b.requires_grad_()
    sigmoid = torch.nn.Sigmoid()
    ta = sigmoid(a)
    print(a)
    print(b)
    print(ta)
    c = torch.cat([ta, b], dim=-1)
    print(c)
    assert True 