/**
 * @file util/math/gaussian.h
 * @author sailing-innocent
 * @date 2023-01-18
 * @brief Geometry and Math Transformation
 */

#include <luisa/dsl/sugar.h>
#include "transform.h"

namespace sail::inno::math {

template<typename Float3_T, typename Float4_T, typename Float3x3_T>
Float3x3_T calc_cov(Float3_T scale, Float4_T qvec) {
	// $\mathbf{R}=\left[\begin{matrix}1-2x^2-2y^2 & 2xy-2rz & 2xz+2ry \\ 2xy+2rz & 1-2x^2-2z^2 & 2yz-2rx \\ 2xz-2ry & 2yz+2rx & 1-2x^2-2y^2 \end{matrix}\right]$
	// LuisaCompute is Col-Major
	Float3x3_T R = R_from_qvec<Float4_T, Float3x3_T>(qvec);
	Float3x3_T S;
	S[0][0] = scale.x;
	S[1][1] = scale.y;
	S[2][2] = scale.z;
	// compute covariance
	// $\Sigma=RSS^TR^T$
	// in source code, it use R^T in reality, but back to true finally
	// ?????
	Float3x3_T M = R * S;
	return M * transpose(M);
}

template<typename Float3_T, typename Float4_T, typename Float3x3_T>
void calc_cov_backward(
	Float3x3_T dL_dSigma,					// input
	Float3_T& dL_dscale, Float4_T& dL_dqvec,// output
	// params
	Float3_T scale, Float4_T qvec) {
	Float3x3_T R = R_from_qvec<Float4_T, Float3x3_T>(qvec);
	Float3x3_T dL_dM = 2.0f * dL_dSigma;
	Float3x3_T dL_dR;
	for (int i = 0; i < 3; i++) {
		dL_dscale[i] = dot(R[i], dL_dM[i]);
		dL_dR[i] = scale[i] * dL_dM[i];
	}
	dL_dqvec = R_from_qvec_backward<Float3_T, Float4_T, Float3x3_T>(dL_dR, qvec, R);
}

template<typename Float4_T, typename Float3x3_T>
Float3x3_T calc_J(Float4_T camera_primitive, Float4_T p_view) {
	auto focal_x = camera_primitive.x;
	auto focal_y = camera_primitive.y;
	auto tan_fov_x = camera_primitive.z;
	auto tan_fov_y = camera_primitive.w;

	auto t = p_view.xyz();
	auto limx = 1.3f * tan_fov_x;
	auto limy = 1.3f * tan_fov_y;

	auto txtz = t.x / t.z;
	auto tytz = t.y / t.z;

	t.x = clamp(txtz, -limx, limx) * t.z;
	t.y = clamp(tytz, -limy, limy) * t.z;

	Float3x3_T J = make_float3x3(
		focal_x / t.z, 0.0f, -(focal_x * t.x) / (t.z * t.z),
		0.0f, focal_y / t.z, -(focal_y * t.y) / (t.z * t.z),
		0.0f, 0.0f, 0.0f);

	return J;
}

// Jacobian for view-space -> ray-space
template<typename Float3_T, typename Float3x3_T>
Float3x3_T J_view2ray(Float3_T u) {
	auto s = sqrt(u.x * u.x + u.y * u.y + u.z * u.z);
	Float3x3_T J = make_float3x3(
		1.0f / u.z, 0.0f, u.x / s,					   // col 1
		0.0f, 1.0f / u.z, u.y / s,					   // col 2
		-u.x / (u.z * u.z), -u.y / (u.z * u.z), u.z / s// col 3
	);
	return J;
}

template<typename Float3_T, typename Float3x3_T, typename Float4x4_T>
Float3_T proj_cov3d_to_cov2d_01(Float3_T p_view, Float3x3_T cov3d, Float4x4_T view_matrix) {
	Float3x3_T J = J_view2ray<Float3_T, Float3x3_T>(p_view);
	Float3x3_T W = make_float3x3(
		view_matrix[0].xyz(),
		view_matrix[1].xyz(),
		view_matrix[2].xyz());
	Float3x3_T T = J * W;
	Float3x3_T cov = transpose(T) * cov3d * T;
	// low pass filter
	cov[0][0] += 1e-5f;
	cov[1][1] += 1e-5f;
	return make_float3(cov[0][0], cov[0][1], cov[1][1]);
}

template<typename Float3_T, typename Float4_T, typename Float3x3_T, typename Float4x4_T>
Float3_T proj_cov3d_to_cov2d(Float4_T p_view, Float4_T camera_primitive, Float3x3_T cov3d, Float4x4_T view_matrix) {
	Float3x3_T J = calc_J<Float4_T, Float3x3_T>(camera_primitive, p_view);

	Float3x3_T W = make_float3x3(
		view_matrix[0].xyz(),
		view_matrix[1].xyz(),
		view_matrix[2].xyz());
	Float3x3_T T = J * W;
	Float3x3_T cov = T * cov3d * transpose(T);
	// low pass filter
	// auto focal = camera_primitive.y;
	cov[0][0] += 0.3f;
	cov[1][1] += 0.3f;
	return make_float3(cov[0][0], cov[0][1], cov[1][1]);
}

template<typename Float3_T, typename Float4_T, typename Float3x3_T, typename Float4x4_T>
void proj_cov3d_to_cov2d_backward(
	Float3_T dL_d_cov_2d,// input
	Float3x3_T& dL_dcov, // output
	Float4_T p_view, Float4_T camera_primitive,
	Float4x4_T view_matrix) {
	Float3x3_T J = calc_J<Float4_T, Float3x3_T>(camera_primitive, p_view);
	Float3x3_T W = make_float3x3(
		view_matrix[0].xyz(),
		view_matrix[1].xyz(),
		view_matrix[2].xyz());
	Float3x3_T T = J * W;

	auto dL_da = dL_d_cov_2d.x;
	auto dL_db = dL_d_cov_2d.y;
	auto dL_dc = dL_d_cov_2d.z;

	// Gradients of loss L w.r.t. each 3D covariance matrix (Vrk) entry,
	// given gradients w.r.t. 2D covariance matrix (diagonal).
	// cov2D = transpose(T) * transpose(Vrk) * T;
	dL_dcov[0][0] = (T[0][0] * T[0][0] * dL_da + T[0][0] * T[1][0] * dL_db + T[1][0] * T[1][0] * dL_dc);
	dL_dcov[1][1] = (T[0][1] * T[0][1] * dL_da + T[0][1] * T[1][1] * dL_db + T[1][1] * T[1][1] * dL_dc);
	dL_dcov[2][2] = (T[0][2] * T[0][2] * dL_da + T[0][2] * T[1][2] * dL_db + T[1][2] * T[1][2] * dL_dc);

	// Gradients of loss L w.r.t. each 3D covariance matrix (Vrk) entry,
	// given gradients w.r.t. 2D covariance matrix (off-diagonal).
	// Off-diagonal elements appear twice --> double the gradient.
	// cov2D = transpose(T) * transpose(Vrk) * T;
	dL_dcov[0][1] = 2 * T[0][0] * T[0][1] * dL_da + (T[0][0] * T[1][1] + T[0][1] * T[1][0]) * dL_db + 2 * T[1][0] * T[1][1] * dL_dc;
	dL_dcov[0][2] = 2 * T[0][0] * T[0][2] * dL_da + (T[0][0] * T[1][2] + T[0][2] * T[1][0]) * dL_db + 2 * T[1][0] * T[1][2] * dL_dc;
	dL_dcov[1][2] = 2 * T[0][2] * T[0][1] * dL_da + (T[0][1] * T[1][2] + T[0][2] * T[1][1]) * dL_db + 2 * T[1][1] * T[1][2];

	// symmetry
	dL_dcov[1][0] = dL_dcov[0][1];
	dL_dcov[2][0] = dL_dcov[0][2];
	dL_dcov[2][1] = dL_dcov[1][2];
}
}// namespace sail::inno::math
