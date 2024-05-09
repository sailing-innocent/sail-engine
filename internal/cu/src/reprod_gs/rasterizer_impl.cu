/**
 * @file rasterizer_impl.cu
 * @brief The Rasterizer Implementation
 * @author sailing-innocent
 * @date 2024-05-09
 */

#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cub/cub.cuh>
#include <cub/device/device_radix_sort.cuh>
#define GLM_FORCE_CUDA
#include <glm/glm.hpp>

#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>

namespace cg = cooperative_groups;

#include "SailCu/reprod_gs/rasterizer_impl.h"
#include "SailCu/reprod_gs/auxiliary.cuh"

namespace sail::cu::reprod_gs {

}// namespace sail::cu::reprod_gs