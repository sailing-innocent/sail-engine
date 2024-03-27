import sys 
from innopy import DiffGSProjectorApp


import torch 
import torch.nn as nn 
from typing import NamedTuple

class _DiffGSProjector(torch.autograd.Function):
    @staticmethod 
    def forward(ctx, xyz, feature, scale, rotq, settings, app):
        ctx.app = app 
        # dpkg settings
        fov_rad = settings.fov_rad
        aspect = settings.aspect
        scale_modifier = settings.scale_modifier
        viewmatrix = settings.viewmatrix
        projmatrix = settings.projmatrix

        sh_degree = settings.sh_degree
        max_sh_degree = settings.max_sh_degree
        campos = settings.campos
        P = xyz.shape[0]

        means_2d = torch.zeros((P, 2), dtype=torch.float32).cuda()
        covs_2d = torch.zeros((P, 3), dtype=torch.float32).cuda()
        depth_features = torch.ones((P, 1), dtype=torch.float32).cuda()
        color_features = torch.ones((P, 3), dtype=torch.float32).cuda()

        app.forward(
            P, sh_degree, max_sh_degree, scale_modifier,
            # input
            xyz.contiguous().data_ptr(), 
            feature.contiguous().data_ptr(), 
            scale.contiguous().data_ptr(), 
            rotq.contiguous().data_ptr(),
            # output
            means_2d.contiguous().data_ptr(),
            covs_2d.contiguous().data_ptr(),
            depth_features.contiguous().data_ptr(),
            color_features.contiguous().data_ptr(),
            # camera
            campos, fov_rad, aspect, viewmatrix, projmatrix
            )
        
        app.sync()

        return means_2d, covs_2d, depth_features, color_features

    @staticmethod 
    def backward(ctx, dL_dmeans_2d, dL_dcovs_2d, dL_ddepth_features, dL_dcolor_features):
        grad = (
            None, 
            None,
            None, 
            None,
            None,
            None
        )
        return grad

class DiffGSProjectorSettings(NamedTuple):
    sh_degree : int
    max_sh_degree: int
    scale_modifier : float
    campos : list
    fov_rad: float
    aspect: float
    viewmatrix : list
    projmatrix : list

class DiffGSProjector(nn.Module):
    def __init__(self):
        super(DiffGSProjector, self).__init__()
        self.app = DiffGSProjectorApp()
        self.app.create(sys.path[-1], "cuda")

    def forward(self, xyz, feature, scale, rotq, settings):
        return _DiffGSProjector.apply(xyz, feature, scale, rotq, settings, self.app)