from lib.torch_ext.sh_processor import SHProcessor, SHProcessorSettings
from module.utils.camera.basic import Camera
from module.utils.torch.sh import eval_sh

import torch 

class GaussianSHProcessor:
    def __init__(self):
        pass

    def process(self, gaussians, camera: Camera):
        campos = torch.from_numpy(camera.info.T.flatten()).float().cuda()
        dirs = gaussians.get_xyz - campos
        dirs = dirs / torch.norm(dirs, dim = -1, keepdim = True)
        return eval_sh(gaussians.max_sh_degree, gaussians.get_features.transpose(1,2), dirs)