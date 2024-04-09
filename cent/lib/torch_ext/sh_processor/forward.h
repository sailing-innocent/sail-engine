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

#ifndef CUDA_SAMPLER_FORWARD_H_INCLUDED
#define CUDA_SAMPLER_FORWARD_H_INCLUDED

#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#define GLM_FORCE_CUDA
#include <glm/glm.hpp>

namespace FORWARD {
// Perform initial steps for each Gaussian prior to rasterization.
void eval_sh(int P, int D, int M,
			 const float* shs,
			 const float* dirs,
			 float* colors,
			 bool* clamped);
}// namespace FORWARD
#endif