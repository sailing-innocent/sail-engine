import sys 
from innopy import DiffGSTileSamplerApp

# DiffGSTileSamplerApp()
# forward
# # num_gaussians
# # width: int 
# # height: int
# # means_2d: N x 2 (X, Y)
# # covs_2d: N  x 3 (C00, C01, C11)
# # depth_features: N x 1 (D)
# # color_features: N x 4 (RGBA)
# # result
# backward

import torch 
import torch.nn as nn 

from typing import NamedTuple

class DiffGSTileSamplerSettings(NamedTuple):
    width: int
    height: int
    fov_rad: float

class _DiffGSTileSampler(torch.autograd.Function):
    @staticmethod
    def forward(ctx, 
                means_2d, covs_2d, depth_features, opacity_features, color_features, 
                settings, app):
        width = settings.width
        height = settings.height
        fov_rad = settings.fov_rad
        result_img = torch.zeros((3, height, width), dtype=torch.float32).cuda()
        P = means_2d.shape[0]
        # print(means_2d)
        # print("covs_2d")
        # print(covs_2d)
        # print(P)
        # print(width)
        # print(height)
        # print(fov_rad)
        app.forward(P, height, width, fov_rad,
                    means_2d.contiguous().data_ptr(), 
                    covs_2d.contiguous().data_ptr(), 
                    depth_features.contiguous().data_ptr(), 
                    opacity_features.contiguous().data_ptr(),
                    color_features.contiguous().data_ptr(), 
                    result_img.contiguous().data_ptr())
        
        ctx.app = app # save for backward
        ctx.width = width
        ctx.height = height
        ctx.num_gaussians = P
        ctx.save_for_backward(covs_2d, opacity_features, color_features)  

        return result_img

    @staticmethod
    def backward(ctx, dL_dtpix):
        P = ctx.num_gaussians
        # print(f"P: {P}")
        # print(f"dL_dtpix: {dL_dtpix.shape}")
        covs_2d, opacity_features, color_features = ctx.saved_tensors

        dL_dmeans_2d = torch.zeros((P, 2), dtype=torch.float32).cuda()
        dL_dcovs_2d = torch.zeros((P, 3), dtype=torch.float32).cuda()
        dL_d_opacity_features = torch.zeros((P, 1), dtype=torch.float32).cuda()
        dL_d_color_features = torch.zeros((P, 3), dtype=torch.float32).cuda()

        ctx.app.backward(dL_dtpix.contiguous().data_ptr(), 
                         covs_2d.contiguous().data_ptr(),
                         opacity_features.contiguous().data_ptr(),
                         color_features.contiguous().data_ptr(),
                         dL_dmeans_2d.contiguous().data_ptr(), 
                         dL_dcovs_2d.contiguous().data_ptr(), 
                         dL_d_opacity_features.contiguous().data_ptr(),
                         dL_d_color_features.contiguous().data_ptr())
        
        # print(dL_d_color_features)
        # print("dL_dcovs_2d")
        # print(dL_dcovs_2d)
        # print(dL_d_opacity_features)
        # print(dL_dmeans_2d)
        grads = (
            dL_dmeans_2d,
            dL_dcovs_2d,
            None, # not for depth
            dL_d_opacity_features,
            dL_d_color_features,
            None, # settings
            None # app
        )
        return grads

class DiffGSTileSampler(nn.Module):
    def __init__(self):
        super(DiffGSTileSampler, self).__init__()
        self.app = DiffGSTileSamplerApp()
        self.app.create(sys.path[-1], "cuda")

    def forward(self, means_2d, covs_2d, depth_features, opacity_features, color_features, settings):
        return _DiffGSTileSampler.apply(means_2d, covs_2d, depth_features, opacity_features, color_features, settings, self.app)