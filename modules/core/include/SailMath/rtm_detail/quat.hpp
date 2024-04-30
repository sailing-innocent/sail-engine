#pragma once
/**
 * @file quat.hpp
 * @brief The RTM Quaternion Wrapper
 * @author sailing-innocent
 * @date 2024-04-30
 */

#include <rtm/quatf.h>
#include <EASTL/array.h>

#include "vector.hpp"
#include "matrix.hpp"

namespace sail {
class Mat4f;
class Vec4f;
class QuatF {
	rtm::quatf __data;
	eastl::array<float, 4> __array;

public:
	QuatF() {
		__data = rtm::quat_identity();
	}
	QuatF(float x, float y, float z, float w) {
		__array[0] = x;
		__array[1] = y;
		__array[2] = z;
		__array[3] = w;
		upload();
	}

	QuatF(rtm::quatf data) : __data(data) {
		download();
	}

	void download() {
		__array[0] = rtm::quat_get_x(__data);
		__array[1] = rtm::quat_get_y(__data);
		__array[2] = rtm::quat_get_z(__data);
		__array[3] = rtm::quat_get_w(__data);
	}

	void upload() {
		__data = rtm::quat_set(__array[0], __array[1], __array[2], __array[3]);
	}

	// getter
	float x() const { return __array[0]; }
	float y() const { return __array[1]; }
	float z() const { return __array[2]; }
	float w() const { return __array[3]; }

	rtm::quatf data() const {
		return __data;
	}

	// operator
	QuatF operator*(const QuatF& rhs) const {
		return QuatF(rtm::quat_mul(__data, rhs.__data));
	}

	QuatF operator-() const {
		return QuatF(rtm::quat_neg(__data));
	}

	friend Vec4f operator*(const Vec4f& lhs, const QuatF& rhs) {
		return Vec4f(rtm::quat_mul_vector3(lhs.data(), rhs.__data));
	}
};

}// namespace sail