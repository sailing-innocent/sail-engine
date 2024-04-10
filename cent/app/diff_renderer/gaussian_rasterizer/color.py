from module.utils.camera.basic import Camera
from lib.torch_ext.gs_cam_sampler import GaussianSampleSettings, GaussianSampler
import numpy as np
import torch 

def create_gaussian_renderer(env_config, config_settings: dict = {}):
    config = GaussianRendererConfig(env_config)
    for key, value in config_settings.items():
        setattr(config, key, value)
    return GaussianRenderer(config)

class GaussianRendererConfig:
    def __init__(self, env_config):
        self._env_config = env_config
        self.white_bkgd = True 

    @property
    def env_config(self):
        return self._env_config

class GaussianRenderer:
    def __init__(self, config: GaussianRendererConfig):
        self.config = config 

    def render(self, camera: Camera, gaussians):
        # logger.info("Rendering with Vanilla Renderer")
        view_mat = torch.tensor(camera.view_matrix.T).float().cuda()
        proj_mat = torch.tensor(camera.full_proj_matrix.T).float().cuda()
        width = camera.info.ResW
        height = camera.info.ResH
        fovx = camera.info.FovX
        fovy = camera.info.FovY
        tanfovx = np.tan(0.5 * fovx)
        tanfovy = np.tan(0.5 * fovy)
        if self.config.white_bkgd:
            bg_color = torch.tensor([1, 1, 1], dtype=torch.float32, device="cuda")
        else:
            bg_color = torch.tensor([0, 0, 0], dtype=torch.float32, device="cuda")

        settings = GaussianSampleSettings(
            image_height = int(height),
            image_width = int(width),
            tanfovx = tanfovx,
            tanfovy = tanfovy,
            bg = bg_color,
            scale_modifier = 1,
            viewmatrix = view_mat,
            projmatrix = proj_mat,
            prefiltered = False,
            debug = False
        )

        rasterizer = GaussianSampler(sample_settings=settings)
        means_3d = gaussians.get_xyz
        screenspace_points = torch.zeros_like(gaussians.get_xyz, dtype=gaussians.get_xyz.dtype, requires_grad=True, device="cuda") + 0
        try:
            screenspace_points.retain_grad()
        except:
            pass
        
        means_2d = screenspace_points
        opacity = gaussians.get_opacity
        scales = gaussians.get_scaling
        rotations = gaussians.get_rotation
        colors = gaussians.get_features

        rendered_image, radii = rasterizer(
            means3D = means_3d,
            means2D = means_2d,
            colors = colors,
            opacities = opacity,
            scales = scales,
            rotations = rotations,
            cov3D_precomp = None 
        )
        return {"render": rendered_image,
                "viewspace_points": screenspace_points,
                "visibility_filter" : radii > 0,
                "radii": radii}