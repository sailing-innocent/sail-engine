/*
 * Copyright (C) 2023, Inria
 * GRAPHDECO research group, https://team.inria.fr/graphdeco
 * All rights reserved.
 *
 * This software is free for non-commercial, research and evaluation use
 * under the terms of the LICENSE.md file.
 *
 * For inquiries contact  george.drettakis@inria.fr
 */

#include <cstdio>
#include <string>
#include <torch/extension.h>
#include <tuple>

std::tuple<int,           // num_rendered
           torch::Tensor, // out_color
           torch::Tensor, // radii
           torch::Tensor, // geom_buffer
           torch::Tensor, // binning buffer
           torch::Tensor, // img_buffer
           torch::Tensor  // sampled background
           >
RasterizeGaussiansCUDA(
    const torch::Tensor &pano, const torch::Tensor &means3D,
    const torch::Tensor &colors, const torch::Tensor &opacity,
    const torch::Tensor &scales, const torch::Tensor &rotations,
    const float scale_modifier, const torch::Tensor &cov3D_precomp,
    const torch::Tensor &dirs, const torch::Tensor &viewmatrix,
    const torch::Tensor &projmatrix, const float tan_fovx, const float tan_fovy,
    const int image_height, const int image_width, const torch::Tensor &sh,
    const int degree, const torch::Tensor &campos, const bool prefiltered,
    const bool debug);

std::tuple<torch::Tensor, // dL_d_means2D
           torch::Tensor, // dL_dcolors
           torch::Tensor, // dL_dopacity
           torch::Tensor, // dL_dmeans3D
           torch::Tensor, // dL_dcov3d
           torch::Tensor, // dL_dsh
           torch::Tensor, // dL_dscales
           torch::Tensor, // dL_drotations
           torch::Tensor  // dL_dpano
           >
RasterizeGaussiansBackwardCUDA(
    const torch::Tensor &background, const torch::Tensor &means3D,
    const torch::Tensor &radii, const torch::Tensor &colors,
    const torch::Tensor &scales, const torch::Tensor &rotations,
    const float scale_modifier, const torch::Tensor &cov3D_precomp,
    const torch::Tensor &dirs, const torch::Tensor &viewmatrix,
    const torch::Tensor &projmatrix, const float tan_fovx, const float tan_fovy,
    const torch::Tensor &dL_dout_color, const torch::Tensor &sh,
    const int degree, const torch::Tensor &campos,
    const torch::Tensor &geomBuffer, const int R,
    const torch::Tensor &binningBuffer, const torch::Tensor &imageBuffer,
    const int pano_h, const int pano_w, const bool debug);

torch::Tensor markVisible(torch::Tensor &means3D, torch::Tensor &viewmatrix,
                          torch::Tensor &projmatrix);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &RasterizeGaussiansCUDA);
  m.def("backward", &RasterizeGaussiansBackwardCUDA);
  m.def("mark_visible", &markVisible);
}