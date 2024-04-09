import pytest 

from lib.torch_ext.sh_processor import SHProcessor, SHProcessorSettings
import torch 

@pytest.mark.current 
def test_sh_processor():
    D = 3
    settings = SHProcessorSettings(sh_degree=D)
    M = 16
    P = 4
    sh = torch.zeros(P, M, 3).float().cuda()
    dirs = torch.ones(P, 3).float().cuda()
    dirs_norm = dirs / torch.norm(dirs, dim=-1, keepdim=True)
    # print(dirs_norm)
    sh_processor = SHProcessor(settings=settings)
    target_color = sh_processor(sh, dirs_norm).detach()
    assert target_color.shape == (P, 3)
    # print(target_color)
    sh[:, :, 0] += 0.1
    sh[:, :, 1] += 0.2
    sh[:, :, 2] += 0.3
    sh.requires_grad = True
    dirs.requires_grad = True
    N_ITER = 300
    N_LOG = 100
    optimizer = torch.optim.Adam([sh, dirs], lr=1e-2)
    for i in range(N_ITER):
        color = sh_processor(sh, dirs_norm)
        # print("color ", color)
        # mse
        loss = torch.mean((color - target_color) ** 2)
        loss.backward()
        with torch.no_grad():
            if i % N_LOG == 0:
                print(f"Loss: {loss.item()}")   
            optimizer.step()
            optimizer.zero_grad()
