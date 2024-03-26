from module.utils.camera.basic import Camera 
from app.diff_renderer.gaussian_projector.orth import GaussianProjector
from app.diff_renderer.gaussian_sampler.inno import GaussianSampler


class GaussianRendererConfig:
    def __init__(self, env_config):
        self.env_config = env_config 

class GaussianRenderer:
    def __init__(self, config, plane=2):
        self.projector = GaussianProjector(plane)
        self.sampler = GaussianSampler()
        pass 

    def render(self, camera: Camera, gaussians, scale_modifier=1.0):
        gaussians_2d = self.projector.project(gaussians)
        img =  self.sampler.sample(gaussians_2d, camera.info.ResW, camera.info.ResH)
        return {
            "render": img
        }
    
def create_gaussian_renderer(env_config):
    config = GaussianRendererConfig(env_config)
    return GaussianRenderer(config)