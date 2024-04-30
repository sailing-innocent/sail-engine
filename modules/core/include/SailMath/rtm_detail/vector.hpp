#pragma once
/**
 * @file vector.hpp
 * @brief The RTM Vector Wrapper
 * @author sailing-innocent
 * @date 2024-04-30
 */

#include <rtm/vector4f.h>
#include <EASTL/array.h>

namespace sail {

class Vec4f {
	rtm::vector4f __data;
	eastl::array<float, 4> __array;

public:
	Vec4f() {
		__data = rtm::vector_zero();
	}
	Vec4f(float x, float y, float z, float w) {
		__array[0] = x;
		__array[1] = y;
		__array[2] = z;
		__array[3] = w;
		upload();
	}

	Vec4f(rtm::vector4f data) : __data(data) {
		download();
	}

	void download() {
		__array[0] = rtm::vector_get_x(__data);
		__array[1] = rtm::vector_get_y(__data);
		__array[2] = rtm::vector_get_z(__data);
		__array[3] = rtm::vector_get_w(__data);
	}

	void upload() {
		__data = rtm::vector_set(__array[0], __array[1], __array[2], __array[3]);
	}

	// getter
	float x() const { return __array[0]; }
	float y() const { return __array[1]; }
	float z() const { return __array[2]; }
	float w() const { return __array[3]; }

	rtm::vector4f data() const {
		return __data;
	}

	// operator
	Vec4f operator+(const Vec4f& rhs) const {
		return Vec4f(rtm::vector_add(__data, rhs.__data));
	}

	Vec4f operator-(const Vec4f& rhs) const {
		return Vec4f(rtm::vector_sub(__data, rhs.__data));
	}

	Vec4f operator*(const Vec4f& rhs) const {
		return Vec4f(rtm::vector_mul(__data, rhs.__data));
	}

	Vec4f operator/(const Vec4f& rhs) const {
		return Vec4f(rtm::vector_div(__data, rhs.__data));
	}
};

// vec3f is a special case of Vec4f
class Vec3f : public Vec4f {
public:
	Vec3f(float x, float y, float z) : Vec4f(x, y, z, 0.0f) {}
};

}// namespace sail