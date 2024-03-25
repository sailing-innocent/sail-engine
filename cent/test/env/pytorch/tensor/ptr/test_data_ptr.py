# manipulate tensor with binding cuda
import pytest 
import torch 

@pytest.mark.current 
def test_dataptr():
    a = torch.zeros(2, dtype=torch.float32).cuda()
    a_ptr = a.contiguous().data_ptr()
    assert True 