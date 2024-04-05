#include "pybind11/pybind11.h"

#include "export_diff_app.h"

namespace sail::ing::py {

int add(int a, int b) {
	return a + b;
}

}// namespace sail::ing::py

PYBIND11_MODULE(ingpy, m) {
	m.doc() = "ing python binding";
	m.def("add", &sail::ing::py::add, "A function which adds two numbers");
	sail::ing::py::export_diff_render_app(m);
}