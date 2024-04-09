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
	torch::Tensor// dL_dsh
	>
EvalSHBackwardCUDA(
	// input
	const torch::Tensor& dL_dcolor,
	// params
	const int P, int D, int M,
	const torch::Tensor& shs,
	const torch::Tensor& colors,
	const torch::Tensor& dirs);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
	m.def("forward", &EvalSHCUDA);
	m.def("backward", &EvalSHBackwardCUDA);
}