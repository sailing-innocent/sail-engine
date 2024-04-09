#include <cstdio>
#include <string>
#include <torch/extension.h>
#include <tuple>

std::tuple<
	torch::Tensor,// colors
	torch::Tensor // geom_buffer
	>
EvalSHCUDA(
	// input
	const torch::Tensor& shs,
	const torch::Tensor& dirs,
	// params
	const int D);

std::tuple<
	torch::Tensor,// dL_dsh
	torch::Tensor // dL_ddir
	>
EvalSHBackwardCUDA(
	// input
	const torch::Tensor& dL_dcolor,
	// params
	const int D,
	const torch::Tensor& shs,
	const torch::Tensor& dirs,
	const torch::Tensor& geom_buffer);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
	m.def("forward", &EvalSHCUDA);
	m.def("backward", &EvalSHBackwardCUDA);
}