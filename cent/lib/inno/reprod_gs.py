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
        view_matrix_arr = raster_settings.viewmatrix
        proj_matrix_arr = raster_settings.projmatrix
        campos_arr = raster_settings.campos
        scale_modifier = raster_settings.scale_modifier
        sh_deg = raster_settings.sh_degree
        max_sh_deg = raster_settings.max_sh_degree
        fov_rad = raster_settings.fov_rad

        app.forward(
            h, w,
            rendered_image.contiguous().data_ptr(),
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
        radii = 1
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
        means_3d, features, opacities, scales, rotations, rendered_image = ctx.saved_tensors

        grad_means_3d = torch.zeros(means_3d.shape, dtype=torch.float32, device="cuda")
        grad_features = torch.zeros(features.shape, dtype=torch.float32, device="cuda")
        # print(grad_features.stride())
        grad_opacities = torch.zeros(opacities.shape, dtype=torch.float32, device="cuda")
        grad_scales = torch.zeros(scales.shape, dtype=torch.float32, device="cuda")
        grad_rotations = torch.zeros(rotations.shape, dtype=torch.float32, device="cuda")

        ctx.app.backward(
            grad_out_color.contiguous().data_ptr(),
            grad_means_3d.contiguous().data_ptr(),
            grad_features.contiguous().data_ptr(),
            grad_opacities.contiguous().data_ptr(),
            grad_scales.contiguous().data_ptr(),
            grad_rotations.contiguous().data_ptr(),
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
        # print(grad_features)

        grads = (
            grad_means_3d,
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
        features,
        opacities,
        scales,
        rotations,
        raster_settings):
        bg_color = torch.tensor([1, 1, 1], dtype=torch.float32, device="cuda")
        rendered_image, radii = _RasterizeGaussians.apply(
            means_3d,
            features,
            opacities,
            scales,
            rotations,
            bg_color,
            raster_settings,
            self._app
        ) 
        return rendered_image, radii