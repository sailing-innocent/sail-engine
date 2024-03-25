import torch 

from lib.reimpl.svox2 import SparseGrid, Camera, Rays, RenderOptions
import numpy as np 

class PlenoxelRenderer:
    def __init__(self):
        pass 

    def render(self, camera, grid):
        c2w = torch.tensor(camera.view_to_world()).float().cuda().contiguous()
        fy = float(
            0.5 * camera.info.resh / np.tan(0.5 * camera.info.FovY)
        )
        fx = float(
            0.5 * camera.info.resw / np.tan(0.5 * camera.info.FovX)
        )
        # print(fx, fy)
        cam = Camera(c2w = c2w,
            fx = fx,
            fy = fy,
            width = camera.info.resw,
            height = camera.info.resh)
        im = grid.volume_render_image(cam, use_kernel=True, return_raylen=False)
        return im 
    
    def render_rays(self, batch_origins, batch_dirs, grid):
        rays = svox2.Rays(batch_origins, batch_dirs)
        rgb_pred = grid.volume_render_fused(rays, rgb_gt,
            beta_loss=0.0, # lambda beta
            sparsity_loss=0.0, # lambda sparsity
            randomize=False)
        return rgb_pred

def create_renderer():
    return PlenoxelRenderer()