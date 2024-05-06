import torch 
from . import dummy_add 
def test_dummy_add():
    N = 10
    a = torch.ones(N).float().cuda()
    b = torch.ones(N).float().cuda()
    c = dummy_add(a, b)
    assert torch.all(c == 2)