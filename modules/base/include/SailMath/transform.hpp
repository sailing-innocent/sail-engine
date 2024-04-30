#pragma once

/**
 * @file transform.hpp
 * @brief Transformation in 3D
 * @author sailing-innocent
 * @date 2024-04-30
 */
#include "math.hpp"

namespace sail {
namespace math {

class Transform3D {
	mat4 __mat4;

public:
	Transform3D() : __mat4(1.0f) {}
	// copy constructor
	Transform3D(const Transform3D& t) : __mat4(t.__mat4) {}
	Transform3D operator=(const Transform3D& t) {
		__mat4 = t.__mat4;
		return *this;
	}
	// move constructor
	Transform3D(Transform3D&& t) : __mat4(std::move(t.__mat4)) {}
	Transform3D operator=(Transform3D&& t) {
		__mat4 = std::move(t.__mat4);
		return *this;
	}

	Transform3D(const mat4& m) : __mat4(m) {}
	Transform3D(mat4&& m) : __mat4(std::move(m)) {}

	// qvv
	Transform3D(const quat& r, const vec3& t, const vec3& s) {
		__mat4 = translate<mat4>(mat4(1.0f), t) * mat4_cast(r) * scale<mat4>(mat4(1.0f), s);
	}
	mat4 matrix() const {
		return __mat4;
	}

	// operator *
	Transform3D operator*(const Transform3D& t) const {
		return Transform3D(__mat4 * t.__mat4);
	}
};

}}// namespace sail::math