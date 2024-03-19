# Simple Orthogonal Projector
import torch 

from module.model.gaussian.model import Gaussians2D
from module.model.gaussian.vanilla import GaussianModel
from module.utils.camera.basic import Camera
from module.utils.torch.sh import eval_sh
from module.utils.torch.transform import qvec2R

class GaussianProjector:
    def __init__(self, plane: int = 3):
        self.plane = plane

    def project(self, gaussians: GaussianModel) -> Gaussians2D:
        xyz = gaussians.get_xyz
        N = xyz.shape[0]

        dummy_dirs = torch.ones((N, 3)).cuda()
        gaussians_2d = Gaussians2D()
        shs = gaussians.get_features
        # N, F, 3 -> N, 3, F
        shs = shs.permute(0, 2, 1)
        gaussians_2d.color_features = eval_sh(gaussians.max_sh_degree, shs, dummy_dirs)
        gaussians_2d.opacity_features = gaussians.get_opacity
        # build sigma 
        # build rotation matrix
        # build translation matrix
        # project
        gaussians_2d.means_2d = torch.zeros((N, 2), dtype=torch.float32).cuda()
        scaling = gaussians.get_scaling
        rot = gaussians.get_rotation
        cov_3d = torch.zeros((N, 3, 3), dtype=torch.float32).cuda()
        cov_3d[:, 0, 0] = scaling[:, 0] ** 2
        cov_3d[:, 1, 1] = scaling[:, 1] ** 2
        cov_3d[:, 2, 2] = scaling[:, 2] ** 2
        R = qvec2R(rot)
        cov_3d = torch.matmul(R, torch.matmul(cov_3d, R.transpose(1, 2)))

        gaussians_2d.covs_2d = 0.0001 * torch.ones((N, 3), dtype=torch.float32).cuda()

        if self.plane == 1: # y-z plane
            gaussians_2d.means_2d[:, 0] = xyz[:, 1]
            gaussians_2d.means_2d[:, 1] = xyz[:, 2]
            gaussians_2d.covs_2d[:, 0] = cov_3d[:, 1, 1]
            gaussians_2d.covs_2d[:, 1] = cov_3d[:, 1, 2]
            gaussians_2d.covs_2d[:, 2] = cov_3d[:, 2, 2]
            
        if self.plane == 2: # x-z plane
            gaussians_2d.means_2d[:, 0] = xyz[:, 0]
            gaussians_2d.means_2d[:, 1] = xyz[:, 2]
            gaussians_2d.covs_2d[:, 0] = cov_3d[:, 0, 0]
            gaussians_2d.covs_2d[:, 1] = cov_3d[:, 0, 2]
            gaussians_2d.covs_2d[:, 2] = cov_3d[:, 2, 2]
            
        if self.plane == 3: # x-y plane
            gaussians_2d.means_2d[:, 0] = xyz[:, 0]
            gaussians_2d.means_2d[:, 1] = xyz[:, 1]
            gaussians_2d.covs_2d[:, 0] = cov_3d[:, 0, 0]
            gaussians_2d.covs_2d[:, 1] = cov_3d[:, 0, 1]
            gaussians_2d.covs_2d[:, 2] = cov_3d[:, 1, 1]

        return gaussians_2d