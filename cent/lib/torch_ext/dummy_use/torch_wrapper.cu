#include <torch/extension.h>
#include "torch_wrapper.h"

#include "SailCu/dummy.h"

torch::Tensor dummy_use_torch(const torch::Tensor& a, const torch::Tensor& b) {
	const int N = a.size(0);
	torch::Tensor c = torch::full({N}, 0, a.options());
	sail::cu::cuda_add(
		a.contiguous().data<int>(),
		b.contiguous().data<int>(),
		c.data<int>(), N);
	return c;
}
