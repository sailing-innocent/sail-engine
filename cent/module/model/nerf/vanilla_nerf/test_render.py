import pytest 

from .helpers import get_embedder, render_rays, run_network, batchify_rays, render 
from .model import VanillaNeRFModel 
import matplotlib.pyplot as plt
import os 
import torch 
import matplotlib.pyplot as plt
from module.utils.camera.numpy import RayCameraNP 
import numpy as np

@pytest.mark.current
def test_model_render():
    # prepare model
    model = VanillaNeRFModel()
    ckpt_file_path = "E:/logs/zzh_nerf_vanilla_lego/200000.tar"
    model.load_ckpt(ckpt_file_path)

    # prepare camera rays
    near=2.
    far=6.    
    camera = RayCameraNP()
    camera.lookat(np.array([3, 3, 3]),np.array([0, 0, 0]))
    camera.flip()
    rays_o, rays_d = camera.rays
    rays_o = torch.from_numpy(rays_o).float().cuda()
    rays_d = torch.from_numpy(rays_d).float().cuda()

    # render 
    with torch.no_grad():
        rgb = model(rays_o, rays_d, near, far)
        rgb_img = rgb.detach().cpu().numpy()
        plt.imshow(rgb_img)
        plt.show()
