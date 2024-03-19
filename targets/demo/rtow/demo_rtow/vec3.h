#pragma once

#include "vector.h"

namespace sail::rtow {
using vec3 = Vector<double, 3>;
using point3 = vec3;// 3D point
using color = vec3; // RGB color

inline vec3 unit_vector(const vec3& v) {
	return normalize(v);
}

}// namespace sail::rtow