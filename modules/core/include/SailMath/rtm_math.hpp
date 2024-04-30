#pragma once
/**
 * @file rtm_math.hpp
 * @brief The Wrapper for RTM
 * @author sailing-innocent
 * @date 2024-04-29
 */

#include "rtm_detail/vector.hpp"
#include "rtm_detail/matrix.hpp"
#include "rtm_detail/quat.hpp"

#include "rtm/qvvf.h"

#include <EASTL/unique_ptr.h>

namespace sail {

class Transform {
	rtm::qvvf __data;
	Vec4f __translation;
	QuatF __rotation;
	Vec4f __scale;

	eastl::unique_ptr<Mat4f> __matrix = nullptr;

public:
	Transform() {
		__data = rtm::qvv_identity();
		__translation = Vec4f(0.0f, 0.0f, 0.0f, 1.0f);
		__rotation = QuatF(0.0f, 0.0f, 0.0f, 1.0f);
		__scale = Vec4f(1.0f, 1.0f, 1.0f, 1.0f);
	}

	Transform(Vec4f translation, QuatF rotation, Vec4f scale) {
		__translation = translation;
		__rotation = rotation;
		__scale = scale;
		upload();
	}

	Transform(rtm::qvvf data) : __data(data) {
		download();
	}

	void download() {
		__translation = Vec4f(__data.translation);
		__rotation = QuatF(__data.rotation);
		__scale = Vec4f(__data.scale);
	}

	void upload() {
		__data = rtm::qvv_set(__translation.data(), __rotation.data(), __scale.data());
	}

	// getter
	Vec4f translation() const { return __translation; }
	QuatF rotation() const { return __rotation; }
	Vec4f scale() const { return __scale; }
	Mat4f matrix() {
		if (__matrix == nullptr) {
			__matrix = eastl::make_unique<Mat4f>(rtm::matrix_from_qvv(__data));
		}
		return *__matrix;
	}

	rtm::qvvf data() const {
		return __data;
	}

	// operator
	Transform operator*(const Transform& rhs) const {
		return Transform(rtm::qvv_mul(__data, rhs.__data));
	}
};

}// namespace sail