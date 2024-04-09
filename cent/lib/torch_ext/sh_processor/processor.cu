#include "processor.h"
#include <cstdio>
#include <cuda_runtime_api.h>
#include <functional>
#include <torch/extension.h>
#include <tuple>

std::function<char*(size_t N)> resizeFunctional(torch::Tensor& t) {
	auto lambda = [&t](size_t N) {
		t.resize_({(long long)N});
		return reinterpret_cast<char*>(t.contiguous().data_ptr());
	};
	return lambda;
}

std::tuple<
	torch::Tensor,// colors
	torch::Tensor // geom_buffer
	>
EvalSHCUDA(
	// input
	const torch::Tensor& shs,
	// params
	const int P, int D, int M,
	const torch::Tensor& dirs) {
	auto float_opts = shs.options().dtype(torch::kFloat32);
	torch::Tensor color = torch::full({P, 3}, 0.0, float_opts);
	torch::Device device(torch::kCUDA);
	torch::TensorOptions options(torch::kByte);
	torch::Tensor geom_buffer = torch::empty({0}, options.device(device));
	std::function<char*(size_t)> geom_func = resizeFunctional(geom_buffer);

	if (P != 0) {
		int M = 0;
		if (shs.size(0) != 0) {
			M = shs.size(1);
		}

		CudaSHProcessor::SHProcessor::forward(
			geom_func,
			P, D, M,
			shs.data_ptr<float>(),
			color.data_ptr<float>(),
			dirs.data_ptr<float>());
	}

	return std::make_tuple(color, geom_buffer);
}

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
	const torch::Tensor& dirs) {
}