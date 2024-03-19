import pytest 
import torch 

@pytest.mark.current
def test_torch_bmm():
    a = torch.tensor([[[1,2],[3,4]],[[5,6],[7,8]]], dtype=torch.float32).cuda()
    a.requires_grad = True
    b = torch.tensor([[[9],[10]],[[15],[16]]], dtype=torch.float32).cuda()
    b.requires_grad = True
    c = torch.bmm(a, b)
    assert torch.all(c == torch.tensor([[[29], [67]], [[171], [233]]], dtype=torch.float32).cuda())
    csum = c.sum()
    csum.backward()
    assert torch.all(a.grad == torch.tensor([[[9, 10], [9, 10]], [[15, 16], [15, 16]]], dtype=torch.float32).cuda())
    assert torch.all(b.grad == torch.tensor([[[4], [6]], [[12], [14]]], dtype=torch.float32).cuda())