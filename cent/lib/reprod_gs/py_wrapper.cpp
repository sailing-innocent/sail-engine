/**
 * @file py_wrapper.cpp
 * @brief The Reproduction of Gaussian Splatting Pybind11 Wrapper
 * @author sailing-innocent
 * @date 2024-05-09
 */

#include <torch/extension.h>
#include "torch_wrapper.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
	m.def("rasterize_gaussians", &RasterizeGaussiansCUDA);
	m.def("rasterize_gaussians_backward", &RasterizeGaussiansBackwardCUDA);
	m.def("mark_visible", &markVisible);
}