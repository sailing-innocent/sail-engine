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

class _DiffGSTileSampler(torch.autograd.Function):
    @staticmethod
    def forward(ctx, 
                means_2d, covs_2d, depth_features, color_features, 
                height, width, app):
        result_img = torch.zeros((3, height, width), dtype=torch.float32).cuda()
        P = means_2d.shape[0]
        app.forward(P, height, width, 
                    means_2d.contiguous().data_ptr(), 
                    covs_2d.contiguous().data_ptr(), 
                    depth_features.contiguous().data_ptr(), 
                    color_features.contiguous().data_ptr(), 
                    result_img.contiguous().data_ptr())
        
        ctx.app = app # save for backward

        return result_img

    @staticmethod
    def backward(ctx, dL_dtpix):
        pass 

class DiffGSTileSampler(nn.Module):
    def __init__(self):
        super(DiffGSTileSampler, self).__init__()
        self.app = DiffGSTileSamplerApp()
        self.app.create(sys.path[-1], "cuda")

    def forward(self, means_2d, covs_2d, depth_features, color_features, height, width):
        return _DiffGSTileSampler.apply(means_2d, covs_2d, depth_features, color_features, height, width, self.app)