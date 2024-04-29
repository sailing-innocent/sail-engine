#pragma once
/**
 * @file transform.hpp
 * @brief The Transform
 * @author sailing-innocent
 * @date 2024-04-29
 */
// 3D Geometry Transform
#include <EASTL/vector.h>
#include <EASTL/unique_ptr.h>
#include "SailMath/type/matrix.hpp"
#include "SailMath/type/vector.hpp"
#include "SailMath/type/quaternion.hpp"

namespace sail {

template<int DIM>
class Transform {
	bool is_homogeneous = false;
	bool m_mat_cached = false;

public:
	Transform();
	virtual ~Transform();

	// Get the matrix of the transform
	Matrix<float, DIM, DIM>& matrix();
	// operator *: transform * transform
	Transform<DIM> operator*(const Transform<DIM>& rhs) const;

protected:
	eastl::vector<Transform> m_sub_transform;
	eastl::unique_ptr<
		Matrix<float, DIM, DIM>>
		mp_mat = nullptr;// matrix to be cached
	eastl::unique_ptr<
		Matrix<float, DIM + 1, DIM + 1>>
		mp_homo_mat = nullptr;// matrix of homogeneous coord
};

template<int DIM>
class ScaleTransform : public Transform<DIM> {
public:
	explicit ScaleTransform(float scale);
	explicit ScaleTransform(const Vector<float, DIM>& scale);
	virtual ~ScaleTransform();

private:
	Vector<float, DIM> m_scale;
};

template<int DIM>
class TranslateTransform : public Transform<DIM> {
public:
	explicit TranslateTransform(const Vector<float, DIM>& translate);
	virtual ~TranslateTransform();

private:
	Vector<float, DIM> m_translate;
};

// variant of RotationConstruct Data

template<int DIM>
class RotateTransform : public Transform<DIM> {
	bool m_axis_angle_cached = false;
	bool m_euler_angle_cached = false;
	bool m_quaternion_cached = false;

public:
	explicit RotateTransform(float angle);
	explicit RotateTransform(const Vector<float, DIM>& axis, float angle);
	explicit RotateTransform(const Vector<float, DIM>& euler_angle);
	explicit RotateTransform(const Quaternion<float>& quaternion);

	virtual ~RotateTransform();

	enum class RotateType : int {
		AXIS_ANGLE,
		EULER_ANGLE,
		QUATERNION
	};

	// Rotation Type Transform
	RotateTransform to_axis_angle(bool cached = false);
	RotateTransform to_euler_angle(bool cached = false);
	RotateTransform to_quaternion(bool cached = false);
};

}// namespace sail