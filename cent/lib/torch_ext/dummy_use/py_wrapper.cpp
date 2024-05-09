#include <torch/extension.h>
#include "torch_wrapper.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
	m.def("dummy_use", &dummy_use_torch);
}
