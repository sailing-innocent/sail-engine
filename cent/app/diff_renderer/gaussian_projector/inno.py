from module.utils.camera.basic import Camera
from module.model.gaussian.model import Gaussians2D, Gaussians3D
from lib.inno.diff_gs_projector import DiffGSProjectorModule, DiffGSProjectorSettings
import torch 

class DiffGSProjector:
    def __init__(self):
        self.module = DiffGSProjectorModule()

    def project(self, gaussians: Gaussians3D, cam: Camera) -> Gaussians2D:
        width = cam.info.ResW
        height = cam.info.ResH

        cam_pos_arr = cam.info.T.flatten().tolist()
        view_matrix_arr = cam.view_matrix.T.flatten().tolist()
        proj_matrix_arr = cam.proj_matrix.T.flatten().tolist()
    
        settings = DiffGSProjectorSettings(
            sh_degree = gaussians.sh_degree,
            max_sh_degree = gaussians.max_sh_degree,
            campos = cam_pos_arr,
            fov_rad = cam.info.FovY,
            aspect = width / height,
            viewmatrix = view_matrix_arr,
            projmatrix = proj_matrix_arr
        )

        means_2d, covs_2d, depth_features, color_features = self.module.forward(
            gaussians.xyz, 
            gaussians.feature, 
            gaussians.scale, 
            gaussians.rotq, 
            settings)
        
        result = Gaussians2D()
        result.means_2d = means_2d
        result.covs_2d = covs_2d
        result.depth_features = depth_features
        result.color_features = color_features
        result.opacity_features = gaussians.opacity_features

        return result