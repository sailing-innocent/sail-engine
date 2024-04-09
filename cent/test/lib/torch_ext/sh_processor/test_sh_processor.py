import pytest 

from lib.torch_ext.sh_processor import SHProcessor, SHProcessorSettings
import torch 

@pytest.mark.current 
def test_sh_processor():
    D = 3
    settings = SHProcessorSettings(sh_degree=D)
    M = 16
    P = 10
    sh = torch.rand(P, M, 3).float().cuda()
    dirs = torch.rand(P, 3).float().cuda()
    dirs_norm = torch.norm(dirs, dim=-1, keepdim=True)
    sh_processor = SHProcessor(settings=settings)
    color = sh_processor(sh, dirs_norm)
    # assert color.shape == (P, 3)