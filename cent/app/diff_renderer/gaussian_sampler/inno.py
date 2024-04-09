import torch 
from module.model.gaussian.model import Gaussians2D

from lib.inno.diff_gs_tile_sampler import DiffGSTileSampler, DiffGSTileSamplerSettings

class GaussianSampler:
    def __init__(self):
        self.sampler = DiffGSTileSampler()

    # Gaussians2D in NDC space -> Image
    def sample(self, gaussians: Gaussians2D, width: int, height: int, fov_rad: float = 1.0, screen_space_points = None) -> torch.Tensor:
        settings = DiffGSTileSamplerSettings(
            width, height, fov_rad)
        
        return self.sampler(
            gaussians.means_2d, 
            gaussians.covs_2d, 
            gaussians.depth_features, 
            gaussians.opacity_features, 
            gaussians.color_features, 
            screen_space_points,
            settings)