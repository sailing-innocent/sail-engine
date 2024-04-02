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
        p_view_hom = p_hom @ view_mat.transpose(0, 1)
        p_view = p_view_hom[:, :3] / (p_view_hom[:, 3].unsqueeze(1) + 1e-6)
        mask = p_view[:, 2] > 0.2

        p_proj = torch.zeros(N, 2).float().cuda()
        p_proj = p_view[:, :2] / (p_view[:, 2].unsqueeze(1) + 1e-6)
        result = Gaussians2D()
        result.means_2d = p_proj
        try:
            result.means_2d.retain_grad()
        except:
            pass
        
        R = qvec2R(gaussians.get_rotation)
        s = gaussians.get_scaling
        S = torch.diag_embed(s)
        cov3d = R @ S @ S @ R.transpose(1,2)
        J = torch.zeros(N, 3, 3).float().cuda()
        J[:, 0, 0] = 1.0 / p_view[:, 2]
        J[:, 1, 1] = 1.0 / p_view[:, 2]
        J[:, 2, 0] = - p_view[:, 0] / (p_view[:, 2] * p_view[:, 2])
        J[:, 2, 1] = - p_view[:, 1] / (p_view[:, 2] * p_view[:, 2]) 
        W = view_mat[0:3, 0:3]
        T = J @ W
        covs_2d = T @ cov3d @ T.transpose(1, 2) 
        result.covs_2d = torch.stack([covs_2d[:, 0, 0], covs_2d[:, 0, 1], covs_2d[:, 1, 1]], dim=1).reshape(N, 3)
        det = covs_2d[:, 0, 0] * covs_2d[:, 1, 1] - covs_2d[:, 0, 1] * covs_2d[:, 0, 1]
        mask = mask & (det > 0.0)

        radii = torch.sqrt(det) * 3.0
        radii[~mask] = 0.0
        result.depth_features = p_view[:, 2].clone()
        cam_pos = torch.from_numpy(cam.info.T).float().cuda()
        dirs =  p_view - cam_pos
        dirs = - dirs / (torch.norm(dirs, dim=1).unsqueeze(1) + 1e-6)
        result.color_features = eval_sh(3, gaussians.get_features.transpose(1,2), dirs) + 0.5
        result.opacity_features = gaussians.get_opacity

        return result, radii, mask