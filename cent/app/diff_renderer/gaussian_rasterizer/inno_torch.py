from module.utils.camera.basic import Camera 
from app.diff_renderer.gaussian_projector.torch import GaussianProjector
from app.diff_renderer.gaussian_sampler.inno import GaussianSampler

import torch 

class GaussianRendererConfig:
    def __init__(self, env_config):
        self.env_config = env_config 

class GaussianRenderer:
    def __init__(self, config):
        self.projector = GaussianProjector()
        self.sampler = GaussianSampler()
        pass 

    def render(self, camera: Camera, gaussians, scale_modifier=1.0):
        gaussians_2d, radii, mask = self.projector.project(
            gaussians, camera, scale_modifier)
        img = self.sampler.sample(
            gaussians_2d, 
            camera.info.ResW, 
            camera.info.ResH,
            camera.info.FovY)

        # print(torch.max(radii))

        return {
            "render": img,
            "viewspace_points": gaussians_2d.means_2d,
            "visibility_filter": mask,
            "radii": radii
        }
    
def create_gaussian_renderer(env_config):
    config = GaussianRendererConfig(env_config)
    return GaussianRenderer(config)