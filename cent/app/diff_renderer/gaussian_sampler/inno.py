import torch 
from module.model.gaussian.model import Gaussians2D

from lib.inno.diff_gs_tile_sampler import DiffGSTileSampler

class GaussianSampler:
    def __init__(self):
        self.sampler = DiffGSTileSampler()

    # Gaussians2D in NDC space -> Image
    def sample(self, gaussians: Gaussians2D, width: int, height: int) -> torch.Tensor:
        return self.sampler(
            gaussians.means_2d, 
            gaussians.covs_2d, 
            gaussians.depth_features, 
            gaussians.opacity_features, 
            gaussians.color_features, 
            height, width)