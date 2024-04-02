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
	Float3x3_T S = make_float3x3(
		scale.x, 0.0f, 0.0f,
		0.0f, scale.y, 0.0f,
		0.0f, 0.0f, scale.z);
	Float3x3_T M = R * S;
	Float3x3_T dL_dM = 2.0f * transpose(M) * dL_dSigma;

	Float3x3_T RT = transpose(R);
	$for(i, 3) {
		dL_dscale[i] = dot(RT[i], dL_dM[i]);
	};

	// TODO: backward qvec
}

template<typename Float4_T, typename Float3x3_T>
inline Float3x3_T calc_J(Float4_T camera_primitive, Float4_T p_view) {
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

	// Float3x3 J = make_float3x3(
	// 	focal_x / t.z, 0.0f, -(focal_x * t.x) / (t.z * t.z),
	// 	0.0f, focal_y / t.z, -(focal_y * t.y) / (t.z * t.z),
	// 	0.0f, 0.0f, 0.0f);

	// consider function p = m(t)
	// $p_x=\frac{f_xt_x}{t_z}$
	// $p_y=\frac{f_yt_y}{t_z}$
	// $p_z=1$
	// Calculate the Jacobian of m(t)
	// J =
	// fx/tz, 0.0,  fx*tx/(tz * tz)
	// 0.0,   fy/tz, fy*ty/(tz * tz)
	// 0.0    0.0,   0.0
	Float3x3_T J = make_float3x3(
		focal_x / t.z, 0.0f, 0.0f,
		0.0f, focal_y / t.z, 0.0f,
		-(focal_x * t.x) / (t.z * t.z), -(focal_y * t.y) / (t.z * t.z), 0.0f);

	return J;
}

// Jacobian for view-space -> ray-space
template<typename Float3_T, typename Float3x3_T>
Float3x3_T J_view2ray(Float3_T u) {
	Float3x3_T J = make_float3x3(
		1.0f / u.z, 0.0f, 0.0f,						// col 1
		0.0f, 1.0f / u.z, 0.0f,						// col 2
		-u.x / (u.z * u.z), -u.y / (u.z * u.z), 0.0f// col 3
	);
	return J;
}

template<typename Float3_T, typename Float3x3_T, typename Float4x4_T>
Float3_T proj_cov3d_to_cov2d_screen(Float3_T p_view, Float3x3_T cov3d, Float4x4_T view_matrix) {
	Float3x3_T J = J_view2ray<Float3_T, Float3x3_T>(p_view);
	Float3x3_T W = make_float3x3(
		view_matrix[0].xyz(),
		view_matrix[1].xyz(),
		view_matrix[2].xyz());
	Float3x3_T T = J * W;
	Float3x3_T cov = T * cov3d * transpose(T);
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

	T = transpose(T);
	dL_dcov[0][0] = (T[0][0] * T[0][0] * dL_da + T[0][0] * T[1][0] * dL_db + T[1][0] * T[1][0] * dL_dc);
	dL_dcov[1][1] = (T[0][1] * T[0][1] * dL_da + T[0][1] * T[1][1] * dL_db + T[1][1] * T[1][1] * dL_dc);
	dL_dcov[2][2] = (T[0][2] * T[0][2] * dL_da + T[0][2] * T[1][2] * dL_db + T[1][2] * T[1][2] * dL_dc);
	dL_dcov[0][1] = 2 * T[0][0] * T[0][1] * dL_da + (T[0][0] * T[1][1] + T[0][1] * T[1][0]) * dL_db + 2 * T[1][0] * T[1][1] * dL_dc;
	dL_dcov[0][2] = 2 * T[0][0] * T[0][2] * dL_da + (T[0][0] * T[1][2] + T[0][2] * T[1][0]) * dL_db + 2 * T[1][0] * T[1][2] * dL_dc;
	dL_dcov[1][2] = 2 * T[0][2] * T[0][1] * dL_da + (T[0][1] * T[1][2] + T[0][2] * T[1][1]) * dL_db + 2 * T[1][1] * T[1][2];
	dL_dcov[1][0] = dL_dcov[0][1];
	dL_dcov[2][0] = dL_dcov[0][2];
	dL_dcov[2][1] = dL_dcov[1][2];
	// asym
}

}// namespace sail::inno::math
