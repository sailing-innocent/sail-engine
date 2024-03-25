import pytest 
import torch 

@pytest.mark.func 
def test_cat():
    a = torch.tensor([1,2,3])
    b = torch.tensor([4,5,6])
    c = torch.cat((a,b))
    assert torch.equal(c, torch.tensor([1,2,3,4,5,6]))

    d = torch.tensor([[1,2,3],[4,5,6]])
    e = torch.tensor([[7,8,9],[10,11,12]])
    f = torch.cat((d,e))
    assert torch.equal(f, torch.tensor([[1,2,3],[4,5,6],[7,8,9],[10,11,12]]))
    g = torch.cat((d,e),axis=1)
    assert torch.equal(g, torch.tensor([[1,2,3,7,8,9],[4,5, 6, 10,11,12]]))

@pytest.mark.func
def test_stack():
    a = torch.tensor([1,2,3])
    b = torch.tensor([4,5,6])
    c = torch.stack((a,b))
    assert torch.equal(c, torch.tensor([[1,2,3],[4,5,6]]))

    d = torch.tensor([[1,2,3],[4,5,6]])
    e = torch.tensor([[7,8,9],[10,11,12]])
    f = torch.stack((d,e))
    assert torch.equal(f, torch.tensor([[[1,2,3],[4,5,6]],[[7,8,9],[10,11,12]]]))
    g = torch.stack((d,e),axis=1)
    assert torch.equal(g, torch.tensor([[[1,2,3],[7,8,9]],[[4,5,6],[10,11,12]]]))

@pytest.mark.func
def test_vstack():
    a = torch.tensor([1,2,3])
    b = torch.tensor([4,5,6])
    c = torch.vstack((a,b))
    assert torch.equal(c, torch.tensor([[1,2,3],[4,5,6]]))

    d = torch.tensor([[1],[2],[3]])
    e = torch.tensor([[4],[5],[6]])
    f = torch.vstack((d,e))
    assert torch.equal(f, torch.tensor([[1],[2],[3],[4],[5],[6]]))