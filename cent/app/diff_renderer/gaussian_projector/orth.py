# Simple Orthogonal Projector
import torch 

from module.model.gaussian.model import Gaussians2D, Gaussians3D
from module.utils.camera.basic import Camera

class DiffGSProjector:
    def __init__(self):
        pass 

    def project(self, gaussians: Gaussians3D, plane: int) -> Gaussians2D:
        gaussians_2d = Gaussians2D(gaussians.N)
        gaussians_2d.color_features = gaussians.color_features
        gaussians_2d.opacity_features = gaussians.opacity_features

        # build sigma 
        # build rotation matrix
        # build translation matrix
        # project

        if plane == 1: # y-z plane
            gaussians_2d.means_2d[:, 0] = gaussians.means_3d[:, 1]
            gaussians_2d.means_2d[:, 1] = gaussians.means_3d[:, 2]

            return gaussians_2d
        
        if plane == 2: # x-z plane
            gaussians_2d.means_2d[:, 0] = gaussians.means_3d[:, 0]
            gaussians_2d.means_2d[:, 1] = gaussians.means_3d[:, 2]

            return gaussians_2d
        
        if plane == 3: # x-y plane
            gaussians_2d.means_2d[:, 0] = gaussians.means_3d[:, 0]
            gaussians_2d.means_2d[:, 1] = gaussians.means_3d[:, 1]

            return gaussians_2d
