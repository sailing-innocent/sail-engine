#include "export_render_app.h"

#include "SailInno/app/dummy_render_app.h"
#include "SailInno/app/dummy_diff_render_app.h"

#include "SailInno/app/reprod_gs_app.h"

// #include "app/render/point_render_app.h"
// #include "app/diff_render/gs/zzh_gs_app.h"
// #include "app/diff_render/gs/split_gs_app.h"
// #include "app/diff_render/light_gs_app.h"

namespace sail::inno::py {

void export_render_app(pybind11::module& m) {
	pybind11::class_<app::DummyRenderApp>(m, "DummyRenderApp")
		.def(pybind11::init<>())
		.def("create", &app::DummyRenderApp::create)
		.def("render_cpu", &app::DummyRenderApp::render_cpu)
		.def("render_cuda", &app::DummyRenderApp::render_cuda);

	// pybind11::class_<app::PointRenderApp>(m, "PointRenderApp")
	// 	.def(pybind11::init<>())
	// 	.def("create", &app::PointRenderApp::create)
	// 	.def("render_cuda", &app::PointRenderApp::render_cuda);
}

void export_diff_render_app(pybind11::module& m) {
	pybind11::class_<app::ReprodGSApp>(m, "ReprodGSApp")
		.def(pybind11::init<>())
		.def("create", &app::ReprodGSApp::create)
		.def("forward", &app::ReprodGSApp::forward)
		.def("backward", &app::ReprodGSApp::backward);

	// pybind11::class_<app::SplitGSApp>(m, "SplitGSApp")
	// 	.def(pybind11::init<>())
	// 	.def("create", &app::SplitGSApp::create)
	// 	.def("forward", &app::SplitGSApp::forward)
	// 	.def("backward", &app::SplitGSApp::backward);

	// pybind11::class_<app::ZZHGaussianSplatterApp>(m, "ZZHGaussianSplatterApp")
	// 	.def(pybind11::init<>())
	// 	.def("create", &app::ZZHGaussianSplatterApp::create)
	// 	.def("forward", &app::ZZHGaussianSplatterApp::forward)
	// 	.def("backward", &app::ZZHGaussianSplatterApp::backward);

	// pybind11::class_<app::LightGaussianSplatterApp>(m, "LightGaussianSplatterApp")
	// .def(pybind11::init<>())
	// .def("create", &app::LightGaussianSplatterApp::create)
	// .def("forward", &app::LightGaussianSplatterApp::forward);

	pybind11::class_<app::DummyDiffRenderApp>(m, "DummyDiffRenderApp")
		.def(pybind11::init<>())
		.def("create", &app::DummyDiffRenderApp::create)
		.def("forward", &app::DummyDiffRenderApp::forward)
		.def("backward", &app::DummyDiffRenderApp::backward);
}

}// namespace sail::inno::py