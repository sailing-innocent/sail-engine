import pytest 
import torch 

@pytest.mark.func
def test_ray_gen():
    W = 2
    H = 2
    i, j = torch.meshgrid(torch.linspace(0, W-1, W), torch.linspace(0, H-1, H), indexing=
    'ij')
    i = i.t()
    j = j.t()
    assert i.shape == (2,2)
    assert j.shape == (2,2)
    assert torch.equal(i, torch.tensor([[0,1],[0,1]]))
    assert torch.equal(j, torch.tensor([[0,0],[1,1]]))

    # 艹学到了，好神奇的玩法
    dirs = torch.stack([(i-0.5)/1, -(j-0.5)/1, -torch.ones_like(i)], -1)
    assert torch.equal(dirs, torch.tensor(
        [[[-0.5,0.5,-1],[0.5,0.5,-1]],
        [[-0.5,-0.5,-1],[0.5,-0.5,-1]]]))

