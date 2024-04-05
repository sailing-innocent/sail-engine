#pragma once

#include "pybind11/pybind11.h"
#include "pybind11/stl.h"

namespace sail::ing::py {

void export_diff_render_app(pybind11::module& m);

}// namespace sail::ing::py