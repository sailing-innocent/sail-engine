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
	const torch::Tensor& dirs,
	// params
	const int D) {
	int P = shs.size(0);
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
			shs.contiguous().data_ptr<float>(),
			dirs.contiguous().data_ptr<float>(),
			P, D, M,
			color.contiguous().data_ptr<float>());
	}

	return std::make_tuple(color, geom_buffer);
}

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
	const torch::Tensor& geom_buffer) {
	const int P = shs.size(0);
	int M = 0;
	if (shs.size(0) != 0) {
		M = shs.size(1);
	}
	auto float_opts = shs.options().dtype(torch::kFloat32);
	torch::Tensor dL_dsh = torch::full({P, D, M}, 0.0, float_opts);
	torch::Tensor dL_ddir = torch::full({P, 3}, 0.0, float_opts);
	if (P != 0) {
		CudaSHProcessor::SHProcessor::backward(
			reinterpret_cast<char*>(geom_buffer.contiguous().data_ptr()),
			dL_dcolor.contiguous().data_ptr<float>(),
			P, D, M,
			shs.contiguous().data_ptr<float>(),
			dirs.contiguous().data_ptr<float>(),
			dL_dsh.contiguous().data_ptr<float>(),
			dL_ddir.contiguous().data_ptr<float>());
	}

	return std::make_tuple(dL_dsh, dL_ddir);
}