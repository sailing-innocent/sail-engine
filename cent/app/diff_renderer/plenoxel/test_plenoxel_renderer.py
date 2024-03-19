import pytest 

from lib.reimpl.svox2 import SparseGrid, Camera, Rays, RenderOptions
from module.utils.camera.numpy import RayCameraNP 
import numpy as np
import matplotlib.pyplot as plt
import torch 

from app.renderer.plenoxel.vanilla import PlenoxelRenderer

@pytest.mark.current 
def test_renderer():
    ckpt = "E:/pretrained/plenoxel/nerf_blender_chair.npz"
    device = "cuda:0"
    grid = SparseGrid.load(ckpt, device=device)

    camera = RayCameraNP("FlipY")
    camera.lookat(np.array([3, 3, 3]),np.array([0, 0, 0]))
    # camera.flip()
    renderer = PlenoxelRenderer()

    assert grid.use_background == False 
    with torch.no_grad():
        im = renderer.render(camera, grid)
        im.clamp_(0.0, 1.0)
        im = im.cpu().numpy()
        # flip
        im = im[::-1, :, :]
        plt.imshow(im)
        plt.show()

@pytest.mark.app
def test_plenoxel_renderer():
    ckpt = "E:/pretrained/plenoxel/nerf_blender_chair.npz"
    device = "cuda:0"
    grid = SparseGrid.load(ckpt, device=device)

    camera = RayCameraNP("FlipY")
    camera.lookat(np.array([3, 3, 3]),np.array([0, 0, 0]))
    # camera.flip()

    assert grid.use_background == False 
    with torch.no_grad():
        c2w = torch.tensor(camera.view_to_world()).float().cuda().contiguous()
        cam = Camera(
            c2w = c2w,
            fx = 1000)
        im = grid.volume_render_image(cam, use_kernel=True, return_raylen=False)
        im.clamp_(0.0, 1.0)
        im = im.cpu().numpy()
        # flip
        im = im[::-1, :, :]
        plt.imshow(im)
        plt.show()