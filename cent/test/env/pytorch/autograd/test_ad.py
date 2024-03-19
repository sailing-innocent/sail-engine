import pytest 
import torch

@pytest.mark.func
def test_autograd_for_scalar():
    a = torch.Tensor([[0,1,2],[3,4,5]])
    a.requires_grad_()
    b = 2 * a
    c = b + 1
    out = c.sum()
    assert out == 36
    out.backward()
    assert not torch.any(a.grad - torch.Tensor([[2.0,2.0,2.0],[2.0,2.0,2.0]]))

@pytest.mark.func
def test_autograd_for_vector():
    a = torch.Tensor([[0,1,2],[3,4,5]])
    a.requires_grad_()
    b = 2 * a
    c = b + 1
    out = c.sum()
    gradient = torch.Tensor([[1.0,1.0,1.0], [0.0,0.0,0.0]])
    b.backward(gradient=gradient)
    assert not torch.any(a.grad - torch.Tensor([[2.0,2.0,2.0],[0.0,0.0,0.0]]))

