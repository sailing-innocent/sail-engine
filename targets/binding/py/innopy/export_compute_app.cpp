#include "export_compute_app.h"

// #include "app/compute/parallel_primitive_app.h"
#include "SailInno/app/diff_gs_projector_app.h"
#include "SailInno/app/diff_gs_tile_sampler_app.h"

namespace sail::inno::py {

void export_parallel_app(pybind11::module& m) {
	// pybind11::class_<app::ParallelPrimitiveApp>(m, "ParallelPrimitiveApp")
	// 	.def(pybind11::init<>())
	// 	.def("create", &app::ParallelPrimitiveApp::create)
	// 	.def("inclusive_scan_float_cpu", &app::ParallelPrimitiveApp::inclusive_scan_float_cpu);
}

void export_gaussian_app(pybind11::module& m) {
	pybind11::class_<app::DiffGSProjectorApp>(m, "DiffGSProjectorApp")
		.def(pybind11::init<>())
		.def("create", &app::DiffGSProjectorApp::create)
		.def("sync", &app::DiffGSProjectorApp::sync)
		.def("forward", &app::DiffGSProjectorApp::forward)
		.def("backward", &app::DiffGSProjectorApp::backward);

	pybind11::class_<app::DiffGSTileSamplerApp>(m, "DiffGSTileSamplerApp")
		.def(pybind11::init<>())
		.def("create", &app::DiffGSTileSamplerApp::create)
		.def("forward", &app::DiffGSTileSamplerApp::forward)
		.def("backward", &app::DiffGSTileSamplerApp::backward);
}

}// namespace sail::inno::py