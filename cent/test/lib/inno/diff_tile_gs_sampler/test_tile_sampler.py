import pytest 

from lib.inno.diff_gs_tile_sampler import DiffGSTileSampler
import torch 
import matplotlib.pyplot as plt

@pytest.mark.app
def test_tile_sampler():
    sampler = DiffGSTileSampler()
    N = 2
    height = 512
    width = 512

    means_2d = torch.zeros((N, 2), dtype=torch.float32).cuda()
    
    means_2d[0, 0] = 0.5
    means_2d[0, 1] = 0.5

    covs_2d = 0.25 * torch.ones((N, 3), dtype=torch.float32).cuda()
    covs_2d[:, 1] = 0


    depth_features = torch.ones((N, 1), dtype=torch.float32).cuda()
    depth_features[0, 0] = 0
    # color_features = torch.stack([black, red], dim=0).cuda()
    color_features = torch.ones((N, 3), dtype=torch.float32).cuda()
    # set red
    color_features[:, 0] = 1
    color_features[:, 1] = 0
    color_features[:, 2] = 0

    opacity_features = torch.ones((N, 1), dtype=torch.float32).cuda()

    # color_features = torch.rand((N, 4), dtype=torch.float32).cuda()

    result_img = sampler.forward(means_2d, covs_2d, depth_features, opacity_features, color_features, height, width)
    
    result_img_np = result_img.cpu().detach().numpy()
    # CHW -> HWC
    result_img_np = result_img_np.transpose(1, 2, 0)
    # flip y
    result_img_np = result_img_np[::-1, :, :]
    plt.imshow(result_img_np)
    plt.show()

    assert True 