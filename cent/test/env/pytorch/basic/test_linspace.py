import pytest 
import torch 

@pytest.mark.current 
def test_linspace():
    x_start = 0
    x_end = 10
    N_steps = 11
    a = torch.linspace(x_start, x_end, steps=N_steps)
    print(a)