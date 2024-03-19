#include "pybind11/pybind11.h"
#include "export_demo.h"

namespace sail::cu::py {

int add(int a, int b) {
	return a + b;
}

}// namespace sail::cu::py

PYBIND11_MODULE(sailcupy, m) {
	m.doc() = "sailcu python binding";
	m.def("add", &sail::cu::py::add, "A function which adds two numbers");

	sail::cu::py::export_demo(m);
}