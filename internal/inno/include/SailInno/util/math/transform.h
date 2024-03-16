/**
 * @file util/math/transform.h
 * @author sailing-innocent
 * @date 2023-01-18
 * @brief Geometry and Math Transformation
 */

#include <luisa/dsl/sugar.h>

namespace sail::inno::math {

template<typename Float4_T>
Float4_T qvec_from_aa(Float4_T axis_angle) {
	Float4_T qvec;
	auto angle = axis_angle.w;
	auto axis = axis_angle.xyz();
	auto s = sin(angle / 2.0f);
	qvec.x = axis.x * s;
	qvec.y = axis.y * s;
	qvec.z = axis.z * s;
	qvec.w = cos(angle / 2.0f);
	return qvec;
}

template<typename Float4_T, typename Float3x3_T>
Float3x3_T R_from_aa(Float4_T axis_angle) {
	Float3x3_T R;
	// TODO: implement
	return R;
}

template<typename Float4_T, typename Float3x3_T>
Float3x3_T R_from_qvec(Float4_T q, bool col_major = true) {
	Float3x3_T R;
	// col 1
	R[0][0] = 1 - 2 * q.y * q.y - 2 * q.z * q.z;
	R[0][1] = 2 * q.x * q.y + 2 * q.z * q.w;
	R[0][2] = 2 * q.x * q.z - 2 * q.y * q.w;
	// col 2
	R[1][0] = 2 * q.x * q.y - 2 * q.z * q.w;
	R[1][1] = 1 - 2 * q.x * q.x - 2 * q.z * q.z;
	R[1][2] = 2 * q.y * q.z + 2 * q.x * q.w;
	// col 3
	R[2][0] = 2 * q.x * q.z + 2 * q.y * q.w;
	R[2][1] = 2 * q.y * q.z - 2 * q.x * q.w;
	R[2][2] = 1 - 2 * q.x * q.x - 2 * q.y * q.y;

	return R;
}

template<typename Float3_T, typename Float4_T, typename Float3x3_T>
Float4_T R_from_qvec_backward(Float3x3_T dL_dR, Float4_T q, Float3x3_T R) {
	Float4_T dL_dq;
	// symmetric
	dL_dq.x = 2 * (q.y * (dL_dR[0][1] + dL_dR[1][0]) +
				   q.z * (dL_dR[0][2] + dL_dR[2][0]) +
				   q.w * (dL_dR[1][2] - dL_dR[2][1])) -
			  2 * q.x * (dL_dR[1][1] + dL_dR[2][2]);

	dL_dq.y = 2 * (q.x * (dL_dR[0][1] + dL_dR[1][0]) +
				   q.y * (dL_dR[1][2] + dL_dR[2][1]) +
				   q.w * (dL_dR[2][0] + dL_dR[0][2])) -
			  2 * q.y * (dL_dR[0][0] + dL_dR[2][2]);

	dL_dq.z = 2 * (q.x * (dL_dR[0][2] + dL_dR[2][0]) +
				   q.y * (dL_dR[1][2] + dL_dR[2][1]) +
				   q.w * (dL_dR[0][1] - dL_dR[1][0])) -
			  2 * q.z * (dL_dR[0][0] + dL_dR[1][1]);

	dL_dq.w = 2 * (q.x * (dL_dR[1][2] - dL_dR[2][1]) +
				   q.y * (dL_dR[2][0] - dL_dR[0][2]) +
				   q.z * (dL_dR[0][1] - dL_dR[1][0]));

	return dL_dq;
}

}// namespace sail::inno::math