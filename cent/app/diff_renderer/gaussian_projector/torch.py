from module.utils.camera.basic import Camera
from module.model.gaussian.model import Gaussians2D
from module.model.gaussian.vanilla import GaussianModel
import torch 

#TODO torch render
class GaussianProjector:
    def __init__(self):
       pass

    def project(self, gaussians: GaussianModel, cam: Camera, scale_modifier = 1.0) -> Gaussians2D:
        view_mat = torch.from_numpy(cam.view_matrix.T).cuda()
        assert view_mat.shape == torch.Size([4, 4])
        v12 = torch.stack(view_mat[0, :], view_mat[1, :])
        assert v12.shape == torch.Size([2, 4])
        v3 = view_mat[2, :]

        xyz = gaussians.get_xyz
        N = xyz.shape[0]
        means_2d = v12 * xyz / (v3 * xyz + 1e-6)
        result = Gaussians2D()
        result.means_2d = means_2d
        result.covs_2d = 0.001 * torch.ones(N, 3).cuda()
        result.depth_features = 1.0 * torch.ones(N, 1).cuda()
        result.color_features = 0.5 * torch.ones(N, 3).cuda()
        result.opacity_features = gaussians.get_opacity

        return result