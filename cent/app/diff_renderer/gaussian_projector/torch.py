# based on torch.bmm
import torch 

from module.model.gaussian.model import Gaussians2D, Gaussians3D
from module.utils.camera.basic import Camera

class DiffGSProjector:
    def __init__(self):
        pass 

    def project(self, gaussians: Gaussians2D, camera: Camera) -> Gaussians3D:
        pass 
        # build sigma 
        # build rotation matrix
        # build translation matrix
        # project