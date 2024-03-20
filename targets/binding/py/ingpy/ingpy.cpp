#include "pybind11/pybind11.h"

namespace sail::ing::py {

int add(int a, int b) {
	return a + b;
}

}// namespace sail::ing::py

PYBIND11_MODULE(ingpy, m) {
	m.doc() = "ing python binding";
	m.def("add", &sail::ing::py::add, "A function which adds two numbers");
}