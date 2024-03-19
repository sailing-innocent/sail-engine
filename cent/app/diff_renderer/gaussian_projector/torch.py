from module.utils.camera.basic import Camera
from module.model.gaussian.model import Gaussians2D
from module.model.gaussian.vanilla import GaussianModel
import torch 
from module.utils.torch.sh import SH2RGB, eval_sh
from module.utils.torch.transform import qvec2R

#TODO torch render
class GaussianProjector:
    def __init__(self):
       pass

    def project(self, gaussians: GaussianModel, cam: Camera, scale_modifier = 1.0) -> Gaussians2D:
        view_mat = torch.from_numpy(cam.view_matrix).float().cuda()
        view_mat._requires_grad = False
        xyz = gaussians.get_xyz
        N = xyz.shape[0]
        p_hom = torch.cat([xyz, torch.ones(N, 1).cuda()], dim=1)
        p_view_hom = torch.matmul(p_hom, view_mat.t())
        p_view = p_view_hom[:, :3] / (p_view_hom[:, 3].unsqueeze(1) + 1e-6)
        p_proj = p_view[:, :2] / (p_view[:, 2].unsqueeze(1) + 1e-6)
        # print(p_proj)
        # p_proj = p_proj * scale_modifier
        result = Gaussians2D()
        result.means_2d = p_proj
        R = qvec2R(gaussians.get_rotation)
        s = gaussians.get_scaling
        S = torch.diag_embed(s)
        cov3d = R.transpose(1,2) @ S @ S @ R
        J = torch.zeros(N, 3, 3).float().cuda()
        p_norm = torch.norm(p_view, dim=1)
        J[:, 0, 0] = 1.0 / p_view[:, 2]
        J[:, 1, 1] = 1.0 / p_view[:, 2]
        J[:, 2, 0] = - p_view[:, 0] / (p_norm)
        J[:, 2, 1] = - p_view[:, 1] / (p_norm)
        W = view_mat[0:3, 0:3]
        T = J @ W
        covs_2d = T.transpose(1, 2)  @ cov3d @ T
        result.covs_2d = torch.stack([covs_2d[:, 0, 0], covs_2d[:, 0, 1], covs_2d[:, 1, 1]], dim=1).reshape(N, 3)
        # result.covs_2d = 0.001 * torch.ones(N, 3).float().cuda()
        # result.covs_2d[:, 1] = 0.0
        result.depth_features = p_view[:, 2].clone()
        # print(result.depth_features)
        # dirs =  p_view @ W.inverse()
        cam_pos = torch.from_numpy(cam.info.T).float().cuda()
        dirs = xyz - cam_pos
        dirs = dirs / (torch.norm(dirs, dim=1).unsqueeze(1) + 1e-6)
        result.color_features = eval_sh(3, gaussians.get_features.transpose(1,2), dirs)
        # print(result.color_features)
        result.opacity_features = gaussians.get_opacity
        # print(result.opacity_features)

        return result