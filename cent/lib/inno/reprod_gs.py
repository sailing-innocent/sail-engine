import sys 
import os 
cwd = os.path.join(os.getcwd(), "../bin/release")
sys.path.append(cwd)
from innopy import ReprodGSApp

from typing import NamedTuple

import torch 
import torch.nn as nn 
import numpy as np 

import matplotlib.pyplot as plt

class GaussianRasterizationSettings(NamedTuple):
    image_height: int
    image_width: int 
    fov_rad: float
    scale_modifier : float
    viewmatrix : list
    projmatrix : list
    sh_degree : int
    max_sh_degree: int
    campos : list
    prefiltered : bool
    debug : bool

class _RasterizeGaussians(torch.autograd.Function):
    @staticmethod 
    def forward(
        ctx,
        means_3d,
        means_2d,
        features,
        opacities,
        scales,
        rotations,
        bg_color,
        raster_settings,
        app
    ):
        w = raster_settings.image_width
        h = raster_settings.image_height

        P = 1
        for i in range(len(means_3d.shape) - 1):
            P *= means_3d.shape[i]

        rendered_image = torch.zeros((3, h, w), dtype=torch.float32).cuda()
        radii = torch.zeros(P, dtype=torch.float32).cuda()

        view_matrix_arr = raster_settings.viewmatrix
        proj_matrix_arr = raster_settings.projmatrix
        campos_arr = raster_settings.campos
        scale_modifier = raster_settings.scale_modifier
        sh_deg = raster_settings.sh_degree
        max_sh_deg = raster_settings.max_sh_degree
        # print("sh_deg: ", sh_deg, "max_sh_deg: ", max_sh_deg)

        fov_rad = raster_settings.fov_rad

        app.forward(
            h, w,
            rendered_image.contiguous().data_ptr(),
            radii.contiguous().data_ptr(),
            P, sh_deg, max_sh_deg,
            means_3d.contiguous().data_ptr(),
            features.contiguous().data_ptr(),
            opacities.contiguous().data_ptr(),
            scales.contiguous().data_ptr(),
            rotations.contiguous().data_ptr(),
            scale_modifier,
            campos_arr,
            fov_rad,
            view_matrix_arr,
            proj_matrix_arr)
        
        ctx.app = app
        ctx.save_for_backward(
            means_3d,
            features,
            opacities,
            scales,
            rotations,
            rendered_image)

        return rendered_image, radii 

    @staticmethod 
    def backward(ctx, grad_out_color, _):
        # print(grad_out_color[:10,:10])
        # psedo_grad = -torch.ones_like(grad_out_color, dtype=torch.float32, device="cuda")

        means_3d, features, opacities, scales, rotations, rendered_image = ctx.saved_tensors
        grad_means_3d = torch.zeros(means_3d.shape, dtype=torch.float32, device="cuda")
        P = means_3d.shape[0]
        grad_means_2d = torch.zeros((P, 2), dtype=torch.float32, device="cuda")
        grad_features = torch.zeros(features.shape, dtype=torch.float32, device="cuda")
        
        # print(grad_features.shape)
        grad_opacities = torch.zeros(opacities.shape, dtype=torch.float32, device="cuda")
        grad_scales = torch.zeros(scales.shape, dtype=torch.float32, device="cuda")
        grad_rotations = torch.zeros(rotations.shape, dtype=torch.float32, device="cuda")

        ctx.app.backward(
            # input
            grad_out_color.contiguous().data_ptr(),
            # psedo_grad.contiguous().data_ptr(),
            # output
            grad_means_3d.contiguous().data_ptr(),
            grad_features.contiguous().data_ptr(),
            grad_opacities.contiguous().data_ptr(),
            grad_scales.contiguous().data_ptr(),
            grad_rotations.contiguous().data_ptr(),
            grad_means_2d.contiguous().data_ptr(),
            # params
            rendered_image.contiguous().data_ptr(),
            means_3d.contiguous().data_ptr(),
            features.contiguous().data_ptr(),
            opacities.contiguous().data_ptr(),
            scales.contiguous().data_ptr(),
            rotations.contiguous().data_ptr()
        )
        # for i in range(grad_out_color.shape[1]):
        #     for j in range(grad_out_color.shape[2]):
        #         print(grad_out_color[:, i, j])
        # print(grad_features[:10])
        # print(grad_opacities[:10])
        # print(grad_means_2d[:10])
        # print(grad_scales[:10])
        # print(grad_rotations[:10])
        # print(grad_means_3d)
        grads = (
            grad_means_3d,
            grad_means_2d,
            grad_features,
            grad_opacities,
            grad_scales,
            grad_rotations,
            None, # bg_color
            None, # raster_settings
            None # app
        )
        return grads

class GaussianRasterizer(nn.Module):
    def __init__(self):
        super().__init__()
        self._app = ReprodGSApp()
        self._app.create(cwd, "cuda")

    def forward(self,
        means_3d,
        means_2d,
        features,
        opacities,
        scales,
        rotations,
        raster_settings):
        bg_color = torch.tensor([1, 1, 1], dtype=torch.float32, device="cuda")
        rendered_image, radii = _RasterizeGaussians.apply(
            means_3d,
            means_2d,
            features,
            opacities,
            scales,
            rotations,
            bg_color,
            raster_settings,
            self._app
        ) 
        return rendered_image, radii