#include <torch/extension.h>
#include "torch_wrapper.h"
#include "kernel.h"

torch::Tensor dummy_add_torch(const torch::Tensor& a, const torch::Tensor& b) {
	const int N = a.size(0);
	torch::Tensor c = torch::full({N}, 0, a.options());
	dummy_add(
		a.contiguous().data<float>(),
		b.contiguous().data<float>(),
		c.contiguous().data<float>(),
		N);
	return c;
}
