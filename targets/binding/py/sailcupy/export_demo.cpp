#include "export_demo.h"
#include "SailCu/demo/dummy_diff_render.h"

namespace sail::cu::py {

void export_demo(pybind11::module& m) {
	pybind11::class_<DummyDiffRender>(m, "DummyDiffRenderApp")
		.def(pybind11::init<>())
		.def("forward", &DummyDiffRender::forward_py)
		.def("backward", &DummyDiffRender::backward_py);
}

}// namespace sail::cu::py