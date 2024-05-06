#include <torch/extension.h>
#include "torch_wrapper.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
	m.def("dummy_add", &dummy_add_torch);
}
