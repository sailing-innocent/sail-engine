#pragma once

#include "vector.h"

namespace ing::rtow {
using vec3 = sail::Vector<double, 3>;
using point3 = vec3;// 3D point
using color = vec3; // RGB color

inline vec3 unit_vector(const vec3& v) {
	return normalize(v);
}

}// namespace ing::rtow