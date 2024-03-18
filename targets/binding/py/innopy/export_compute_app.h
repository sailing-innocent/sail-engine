#pragma once

#include "pybind11/pybind11.h"
#include "pybind11/stl.h"

namespace sail::inno::py {

void export_parallel_app(pybind11::module& m);
void export_gaussian_app(pybind11::module& m);

}// namespace sail::inno::py