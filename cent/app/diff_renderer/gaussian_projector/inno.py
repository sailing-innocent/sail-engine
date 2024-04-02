from module.utils.camera.basic import Camera
from module.model.gaussian.model import Gaussians2D
from module.model.gaussian.vanilla import GaussianModel
from lib.inno.diff_gs_projector import DiffGSProjector, DiffGSProjectorSettings

class GaussianProjector:
    def __init__(self):
        self.projector = DiffGSProjector()

    def project(self, gaussians: GaussianModel, cam: Camera, scale_modifier = 1.0) -> Gaussians2D:
        width = cam.info.ResW
        height = cam.info.ResH

        cam_pos_arr = cam.info.T.flatten().tolist()
        view_matrix_arr = cam.view_matrix.T.flatten().tolist()

        settings = DiffGSProjectorSettings(
            sh_degree = gaussians.active_sh_degree,
            max_sh_degree = gaussians.max_sh_degree,
            scale_modifier=scale_modifier,
            campos = cam_pos_arr,
            viewmatrix = view_matrix_arr
        )

        means_2d, covs_2d, depth_features, color_features = self.projector(
            gaussians.get_xyz, 
            gaussians.get_features, 
            gaussians.get_scaling, 
            gaussians.get_rotation, 
            settings)
        
        result = Gaussians2D()
        result.means_2d = means_2d
        result.covs_2d = covs_2d
        result.depth_features = depth_features
        result.color_features = color_features
        result.opacity_features = gaussians.get_opacity

        return result