from module.utils.camera.basic import Camera
from lib.inno.reprod_gs import GaussianRasterizationSettings, GaussianRasterizer

import torch 

rasterizer = GaussianRasterizer() 

def create_gaussian_renderer(env_config):
    config = GaussianRendererConfig(env_config)
    return GaussianRenderer(config)

class GaussianRendererConfig:
    def __init__(self, env_config):
        self.env_config = env_config

class GaussianRenderer:
    def __init__(self, config: GaussianRendererConfig):
        self.config = config

    def render(self, camera: Camera, gaussians, scale_modifier=1.0):
        # LC is col-major but numpy is row-major
        view_mat = camera.view_matrix.T.flatten().tolist()
        proj_mat = camera.proj_matrix.T.flatten().tolist()
        width = camera.info.ResW
        height = camera.info.ResH
        fovy = camera.info.FovY

        campos = camera.info.T.flatten().tolist()
        raster_settings = GaussianRasterizationSettings(
            image_height = int(height),
            image_width = int(width),
            fov_rad = fovy,
            scale_modifier = scale_modifier,
            viewmatrix = view_mat,
            projmatrix = proj_mat,
            sh_degree = gaussians.active_sh_degree,
            max_sh_degree = gaussians.max_sh_degree,
            campos = campos,
            prefiltered = False,
            debug = False
        )

        means_3d = gaussians.get_xyz
        P = means_3d.shape[0]
        # just a place holder, requires its grad for trick
        screenspace_points = torch.zeros((P, 2), dtype=torch.float32, requires_grad=True, device="cuda")
        # try:
        #     screenspace_points.retain_grad()
        # except:
        #     pass
        opacity = gaussians.get_opacity
        scales = gaussians.get_scaling
        rotations = gaussians.get_rotation
        features = gaussians.get_features

        rendered_image, radii = rasterizer(
            means_3d = means_3d,
            means_2d = screenspace_points,
            features = features,
            opacities = opacity,
            scales = scales,
            rotations = rotations,
            raster_settings = raster_settings
        )

        return {
            "render": rendered_image,
            "viewspace_points": screenspace_points,
            "visibility_filter": radii > 0,
            "radii": radii}