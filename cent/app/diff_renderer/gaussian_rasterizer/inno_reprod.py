from module.utils.camera.basic import Camera
from lib.inno.reprod_gs import GaussianRasterizationSettings, GaussianRasterizer

def create_gaussian_renderer(env_config):
    config = GaussianRendererConfig(env_config)
    return GaussianRenderer(config)

class GaussianRendererConfig:
    def __init__(self, env_config):
        self._env_config = env_config

class GaussianRenderer:
    def __init__(self, config: GaussianRendererConfig):
        self.config = config
        self.rasterizer = GaussianRasterizer() 

    def render_scene(self, scene, cam, scale_modifier=0.1):
        width = cam.info.ResW
        height = cam.info.ResH

        cam_pos_arr = cam.info.T.flatten().tolist()
        view_matrix_arr = cam.view_matrix.T.flatten().tolist()
        proj_matrix_arr = cam.proj_matrix.T.flatten().tolist()
        N = scene.n_points
        xyz = scene.xyz_torch()
        opacity = scene.opacity_torch()
        color = scene.color_torch()
        scales = scene.scale_torch()
        rots = scene.rot_torch()

        raster_settings = GaussianRasterizationSettings(
            image_height = int(height),
            image_width = int(width),
            fov_rad = cam.info.FovY,
            scale_modifier = scale_modifier,
            viewmatrix = view_matrix_arr,
            projmatrix = proj_matrix_arr,
            sh_degree = -1,
            max_sh_degree = 0,
            campos = cam_pos_arr,
            prefiltered = False,
            debug = False
        )
        rendered_image, radii = self.rasterizer(
            means_3d = xyz,
            features = color,
            opacities = opacity,
            scales = scales,
            rotations = rots,
            raster_settings = raster_settings
        )

        return rendered_image

    def render(self, camera: Camera, gaussians, scale_modifier=1.0):
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
        opacity = gaussians.get_opacity
        scales = gaussians.get_scaling
        rotations = gaussians.get_rotation
        features = gaussians.get_features

        rendered_image, radii = self.rasterizer(
            means_3d = means_3d,
            features = features,
            opacities = opacity,
            scales = scales,
            rotations = rotations,
            raster_settings = raster_settings
        )
        return {"render": rendered_image}