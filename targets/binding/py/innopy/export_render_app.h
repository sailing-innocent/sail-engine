#include "pybind11/pybind11.h"
#include "pybind11/stl.h"

namespace inno::py
{

void export_render_app(pybind11::module& m);
void export_diff_render_app(pybind11::module& m);

} // namespace inno::py