#pragma once
/**
 * @file matrix.hpp
 * @brief The RTM Matrix Wrapper
 * @author sailing-innocent
 * @date 2024-04-30
 */
#include <rtm/matrix3x3f.h>
#include <rtm/matrix4x4f.h>
// from quaternion
#include <EASTL/array.h>

#include "vector.hpp"
#include "quat.hpp"

namespace sail {
class QuatF;
class Vec4f;

class Mat4f {
	rtm::matrix4x4f __data;
	eastl::array<float, 16> __array;

public:
	Mat4f() {
		__data = rtm::matrix_identity();
	}

	explicit Mat4f(float x0, float x1, float x2, float x3,
				   float y0, float y1, float y2, float y3,
				   float z0, float z1, float z2, float z3,
				   float w0, float w1, float w2, float w3) {
		__array[0] = x0;
		__array[1] = x1;
		__array[2] = x2;
		__array[3] = x3;
		__array[4] = y0;
		__array[5] = y1;
		__array[6] = y2;
		__array[7] = y3;
		__array[8] = z0;
		__array[9] = z1;
		__array[10] = z2;
		__array[11] = z3;
		__array[12] = w0;
		__array[13] = w1;
		__array[14] = w2;
		__array[15] = w3;
		upload();
	}

	Mat4f(rtm::matrix4x4f data) : __data(data) {
		download();
	}
	Mat4f(rtm::matrix3x4f data) {
		auto w_axis = rtm::vector_set(0.0f, 0.0f, 0.0f, 1.0f);
		__data = rtm::matrix_set(data.x_axis, data.y_axis, data.z_axis, w_axis);
		download();
	}

	void download() {
		auto x_axis = __data.x_axis;
		auto y_axis = __data.y_axis;
		auto z_axis = __data.z_axis;
		auto w_axis = __data.w_axis;

		__array[0] = rtm::vector_get_x(x_axis);
		__array[1] = rtm::vector_get_y(x_axis);
		__array[2] = rtm::vector_get_z(x_axis);
		__array[3] = rtm::vector_get_w(x_axis);

		__array[4] = rtm::vector_get_x(y_axis);
		__array[5] = rtm::vector_get_y(y_axis);
		__array[6] = rtm::vector_get_z(y_axis);
		__array[7] = rtm::vector_get_w(y_axis);

		__array[8] = rtm::vector_get_x(z_axis);
		__array[9] = rtm::vector_get_y(z_axis);
		__array[10] = rtm::vector_get_z(z_axis);
		__array[11] = rtm::vector_get_w(z_axis);

		__array[12] = rtm::vector_get_x(w_axis);
		__array[13] = rtm::vector_get_y(w_axis);
		__array[14] = rtm::vector_get_z(w_axis);
		__array[15] = rtm::vector_get_w(w_axis);
	}

	void upload() {
		__data.x_axis = rtm::vector_set(__array[0], __array[1], __array[2], __array[3]);
		__data.y_axis = rtm::vector_set(__array[4], __array[5], __array[6], __array[7]);
		__data.z_axis = rtm::vector_set(__array[8], __array[9], __array[10], __array[11]);
		__data.w_axis = rtm::vector_set(__array[12], __array[13], __array[14], __array[15]);
	}

	rtm::matrix4x4f data() const {
		return __data;
	}

	// getter
	float operator()(int i, int j) const {
		// col major
		return __array[i + j * 4];
	}

	// operator
	Mat4f operator*(const Mat4f& rhs) const {
		return Mat4f(rtm::matrix_mul(__data, rhs.__data));
	}

	friend Vec4f operator*(const Vec4f& lhs, const Mat4f& rhs) {
		return Vec4f(rtm::matrix_mul_vector(lhs.data(), rhs.__data));
	}
};

class Mat3f {
	rtm::matrix3x3f __data;
	eastl::array<float, 9> __array;

public:
	Mat3f() {
		__data = rtm::matrix_identity();
	}

	Mat3f(rtm::matrix3x3f data) : __data(data) {
		download();
	}

	void download() {
		// TODO
	}

	void upload() {
		// TODO
	}

	rtm::matrix3x3f data() const {
		return __data;
	}

	// getter
	float element(int i, int j) const {
		return __array[i * 3 + j];
	}

	// operator
	Mat3f operator*(const Mat3f& rhs) const {
		return Mat3f(rtm::matrix_mul(__data, rhs.__data));
	}
};

}// namespace sail