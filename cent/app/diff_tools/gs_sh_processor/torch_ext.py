from lib.torch_ext.sh_processor import SHProcessor, SHProcessorSettings
from module.utils.camera.basic import Camera

import torch 

class GaussianSHProcessor:
    def __init__(self):
        pass

    def process(self, gaussians, camera: Camera):
        sh = gaussians.get_features
        campos = torch.from_numpy(camera.info.T.flatten()).float().cuda()
        dirs = gaussians.get_xyz - campos
        dirs = dirs / torch.norm(dirs, dim = 1, keepdim = True)
        sh_settings = SHProcessorSettings(
            sh_degree = gaussians.active_sh_degree,
        )
        sh_processor = SHProcessor(sh_settings)
        return sh_processor(sh, dirs)