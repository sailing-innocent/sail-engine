import pytest 
from . import dummy_use
import torch 

@pytest.mark.current 
def test_dummy_use():
    a = torch.tensor([1, 2, 3], dtype=torch.int32).cuda()
    b = torch.tensor([4, 5, 6], dtype=torch.int32).cuda()
    c = dummy_use(a, b)
    print(c)
    assert True 