import pytest 

from module.model.nvs.vanilla_nerf.model import VanillaNeRFModel 
from app.renderer.nerf.vanilla import NeRFRenderer
import matplotlib.pyplot as plt
from module.utils.camera.numpy import RayCameraNP 
import numpy as np

@pytest.mark.current 
def test_simple_render():
    # prepare model 
    model = VanillaNeRFModel()
    ckpt_file_path = "E:/logs/zzh_nerf_vanilla_lego/200000.tar"
    model.load_ckpt(ckpt_file_path)

    # prepare camera
    near=2.
    far=6.    
    camera = RayCameraNP()
    camera.lookat(np.array([3, 3, 3]),np.array([0, 0, 0]))
    camera.flip()

    # renderer
    renderer = NeRFRenderer()
    rgb_img = renderer.render(camera, model, near, far)
    plt.imshow(rgb_img)
    plt.show()