#include "export_diff_app.h"

#include "SailCu/demo/diff_pano_sampler.h"
#include "SailCu/demo/dummy_diff_render.h"

namespace sail::ing::py {

void export_diff_render_app(pybind11::module& m) {
	pybind11::class_<sail::cu::DummyDiffRender>(m, "DummyDiffRender")
		.def(pybind11::init<>())
		.def("forward", &sail::cu::DummyDiffRender::forward_py)
		.def("backward", &sail::cu::DummyDiffRender::backward_py);
	pybind11::class_<sail::cu::DiffPanoSampler>(m, "DiffPanoSampler")
		.def(pybind11::init<>())
		.def("forward", &sail::cu::DiffPanoSampler::forward_py)
		.def("backward", &sail::cu::DiffPanoSampler::backward_py);
}

}// namespace sail::ing::py