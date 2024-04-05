from module.utils.camera.basic import Camera
from lib.torch_ext.pano_gs import GaussianRasterizationSettings, GaussianRasterizer

import numpy as np
import torch 
from loguru import logger 

def create_gaussian_renderer(env_config):
    config = GaussianRendererConfig(env_config)
    return GaussianRenderer(config)

class GaussianRendererConfig:
    def __init__(self, env_config):
        self._env_config = env_config

    @property
    def env_config(self):
        return self._env_config

class GaussianRenderer:
    def __init__(self, config: GaussianRendererConfig):
        self.config = config 
        logger.info("Using Pano GS Renderer")

    def render(self, camera: Camera, gaussians, pano = None):
        _, dirs = camera.rays
        dirs = torch.from_numpy(dirs).float().cuda()
        view_mat = torch.tensor(camera.view_matrix.T).float().cuda()
        proj_mat = torch.tensor(camera.full_proj_matrix.T).float().cuda()

        width = camera.info.ResW
        height = camera.info.ResH
        fovx = camera.info.FovX
        fovy = camera.info.FovY
        tanfovx = np.tan(0.5 * fovx)
        tanfovy = np.tan(0.5 * fovy)
        campos = torch.from_numpy(camera.info.T).float().cuda()
        
        if pano == None:
            pano = torch.ones((3, 2048, 4096)).float().cuda()
            
        raster_settings = GaussianRasterizationSettings(
            image_height = int(height),
            image_width = int(width),
            tanfovx = tanfovx,
            tanfovy = tanfovy,
            scale_modifier = 1,
            dirs = dirs,
            viewmatrix = view_mat,
            projmatrix = proj_mat,
            sh_degree = gaussians.active_sh_degree,
            campos = campos,
            prefiltered = False,
            debug = False
        )

        rasterizer = GaussianRasterizer(raster_settings=raster_settings)

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

        shs = gaussians.get_features

        colors_precomp = None
        
        rendered_image, radii = rasterizer(
            means3D = means_3d,
            means2D = means_2d,
            shs = shs,
            colors_precomp = colors_precomp,
            opacities = opacity,
            scales = scales,
            rotations = rotations,
            pano = pano,
            cov3D_precomp = None 
        )

        return {"render": rendered_image,
                "viewspace_points": screenspace_points,
                "visibility_filter" : radii > 0,
                "radii": radii}