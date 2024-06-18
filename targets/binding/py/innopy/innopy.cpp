#include "pybind11/pybind11.h"
#include "export_render_app.h"
#include "export_compute_app.h"

namespace inno::py {

int add(int a, int b) {
	return a + b;
}

}// namespace inno::py

PYBIND11_MODULE(innopy, m) {
	m.doc() = "inno python binding";
	m.def("add", &inno::py::add, "A function which adds two numbers");
	// sail::inno::py::export_render_app(m);
	// sail::inno::py::export_diff_render_app(m);
	// sail::inno::py::export_parallel_app(m);
	// sail::inno::py::export_gaussian_app(m);
}