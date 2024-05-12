import pytest
import torch 

@pytest.mark.func
def test_unsqueeze():
    a = torch.Tensor([[0,1,2],[3,4,5]])
    assert a.shape == (2,3)
    b = a.unsqueeze(0)
    assert b.shape == (1,2,3)
    c = torch.Tensor([[1,2],[3,4],[5,6]])
    assert c.shape == (3,2)

@pytest.mark.func
def test_expand():
    a = torch.Tensor([[1],[2],[3]])
    assert a.shape == (3,1)
    b = a.expand(3,4)
    assert b.shape == (3,4)
    c = a.expand(-1, 2) # -1 means not change
    assert c.shape == (3,2)
    
    # d = a.expand(4,-1) # error, the expanded size of tensor (4) must match the existing size (3)
@pytest.mark.func 
def test_order():
    # pytorch is the row order first
    a = torch.Tensor([[1,2,3],[4,5,6]])
    b = torch.Tensor([[1,2],[3,4],[5,6]])
    c = a.mm(b)
    assert c.shape == (2,2)

@pytest.mark.func
def test_reshape():
    a = torch.Tensor([[1,2,3],[4,5,6]])
    assert a.shape == (2,3)
    b = a.flatten()
    assert b.shape == (6,)
    c = b.reshape(2,3)
    assert c.shape == (2,3)
    assert torch.equal(a, c)

