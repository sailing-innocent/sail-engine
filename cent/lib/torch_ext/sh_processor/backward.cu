#include "backward.h"
#include "auxiliary.h"
#include "config.h"
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>

#include <cstdio>
namespace cg = cooperative_groups;

// Backward pass for conversion of spherical harmonics to RGB for
// each Gaussian.
__device__ void computeColorFromSH(
	int idx, int deg, int max_coeffs,
	const glm::vec3* dirs,
	const float* shs,
	const bool* clamped,
	const glm::vec3* dL_dcolor,
	glm::vec3* dL_ddirs,
	glm::vec3* dL_dshs) {
	// Compute intermediate values, as it is done during forward
	glm::vec3 dir = dirs[idx];// dir is normalized
	glm::vec3* sh = ((glm::vec3*)shs) + idx * max_coeffs;

	// Use PyTorch rule for clamping: if clamping was applied,
	// gradient becomes 0.
	glm::vec3 dL_dRGB = dL_dcolor[idx];
	dL_dRGB.x *= clamped[3 * idx + 0] ? 0 : 1;
	dL_dRGB.y *= clamped[3 * idx + 1] ? 0 : 1;
	dL_dRGB.z *= clamped[3 * idx + 2] ? 0 : 1;

	glm::vec3 dRGBdx(0, 0, 0);
	glm::vec3 dRGBdy(0, 0, 0);
	glm::vec3 dRGBdz(0, 0, 0);
	float x = dir.x;
	float y = dir.y;
	float z = dir.z;

	// Target location for this Gaussian to write SH gradients to
	glm::vec3* dL_dsh = dL_dshs + idx * max_coeffs;

	// No tricks here, just high school-level calculus.
	float dRGBdsh0 = SH_C0;
	dL_dsh[0] = dRGBdsh0 * dL_dRGB;
	if (deg > 0) {
		float dRGBdsh1 = -SH_C1 * y;
		float dRGBdsh2 = SH_C1 * z;
		float dRGBdsh3 = -SH_C1 * x;
		dL_dsh[1] = dRGBdsh1 * dL_dRGB;
		dL_dsh[2] = dRGBdsh2 * dL_dRGB;
		dL_dsh[3] = dRGBdsh3 * dL_dRGB;

		dRGBdx = -SH_C1 * sh[3];
		dRGBdy = -SH_C1 * sh[1];
		dRGBdz = SH_C1 * sh[2];

		if (deg > 1) {
			float xx = x * x, yy = y * y, zz = z * z;
			float xy = x * y, yz = y * z, xz = x * z;

			float dRGBdsh4 = SH_C2[0] * xy;
			float dRGBdsh5 = SH_C2[1] * yz;
			float dRGBdsh6 = SH_C2[2] * (2.f * zz - xx - yy);
			float dRGBdsh7 = SH_C2[3] * xz;
			float dRGBdsh8 = SH_C2[4] * (xx - yy);
			dL_dsh[4] = dRGBdsh4 * dL_dRGB;
			dL_dsh[5] = dRGBdsh5 * dL_dRGB;
			dL_dsh[6] = dRGBdsh6 * dL_dRGB;
			dL_dsh[7] = dRGBdsh7 * dL_dRGB;
			dL_dsh[8] = dRGBdsh8 * dL_dRGB;

			dRGBdx += SH_C2[0] * y * sh[4] + SH_C2[2] * 2.f * -x * sh[6] + SH_C2[3] * z * sh[7] + SH_C2[4] * 2.f * x * sh[8];
			dRGBdy += SH_C2[0] * x * sh[4] + SH_C2[1] * z * sh[5] + SH_C2[2] * 2.f * -y * sh[6] + SH_C2[4] * 2.f * -y * sh[8];
			dRGBdz += SH_C2[1] * y * sh[5] + SH_C2[2] * 2.f * 2.f * z * sh[6] + SH_C2[3] * x * sh[7];

			if (deg > 2) {
				float dRGBdsh9 = SH_C3[0] * y * (3.f * xx - yy);
				float dRGBdsh10 = SH_C3[1] * xy * z;
				float dRGBdsh11 = SH_C3[2] * y * (4.f * zz - xx - yy);
				float dRGBdsh12 = SH_C3[3] * z * (2.f * zz - 3.f * xx - 3.f * yy);
				float dRGBdsh13 = SH_C3[4] * x * (4.f * zz - xx - yy);
				float dRGBdsh14 = SH_C3[5] * z * (xx - yy);
				float dRGBdsh15 = SH_C3[6] * x * (xx - 3.f * yy);
				dL_dsh[9] = dRGBdsh9 * dL_dRGB;
				dL_dsh[10] = dRGBdsh10 * dL_dRGB;
				dL_dsh[11] = dRGBdsh11 * dL_dRGB;
				dL_dsh[12] = dRGBdsh12 * dL_dRGB;
				dL_dsh[13] = dRGBdsh13 * dL_dRGB;
				dL_dsh[14] = dRGBdsh14 * dL_dRGB;
				dL_dsh[15] = dRGBdsh15 * dL_dRGB;

				dRGBdx += (SH_C3[0] * sh[9] * 3.f * 2.f * xy +
						   SH_C3[1] * sh[10] * yz +
						   SH_C3[2] * sh[11] * -2.f * xy +
						   SH_C3[3] * sh[12] * -3.f * 2.f * xz +
						   SH_C3[4] * sh[13] * (-3.f * xx + 4.f * zz - yy) +
						   SH_C3[5] * sh[14] * 2.f * xz +
						   SH_C3[6] * sh[15] * 3.f * (xx - yy));

				dRGBdy += (SH_C3[0] * sh[9] * 3.f * (xx - yy) +
						   SH_C3[1] * sh[10] * xz +
						   SH_C3[2] * sh[11] * (-3.f * yy + 4.f * zz - xx) +
						   SH_C3[3] * sh[12] * -3.f * 2.f * yz +
						   SH_C3[4] * sh[13] * -2.f * xy +
						   SH_C3[5] * sh[14] * -2.f * yz +
						   SH_C3[6] * sh[15] * -3.f * 2.f * xy);

				dRGBdz += (SH_C3[1] * sh[10] * xy +
						   SH_C3[2] * sh[11] * 4.f * 2.f * yz +
						   SH_C3[3] * sh[12] * 3.f * (2.f * zz - xx - yy) +
						   SH_C3[4] * sh[13] * 4.f * 2.f * xz +
						   SH_C3[5] * sh[14] * (xx - yy));
			}
		}
	}

	// The view direction is an input to the computation. View direction
	// is influenced by the Gaussian's mean, so SHs gradients
	// must propagate back into 3D position.
	dL_ddirs[idx] = glm::vec3(glm::dot(dRGBdx, dL_dRGB), glm::dot(dRGBdy, dL_dRGB), glm::dot(dRGBdz, dL_dRGB));
}
template<int C>
__global__ void eval_sh_CUDA(
	int P, int D, int M,
	const float* shs,
	const float* dirs,
	const bool* clamped,
	// input
	const float* dL_dcolor,
	// output
	float* dL_dsh,
	float* dL_ddir) {

	auto idx = cg::this_grid().thread_rank();
	if (idx >= P) return;
	computeColorFromSH(idx, D, M,
					   (glm::vec3*)dirs,
					   shs,
					   clamped,
					   (glm::vec3*)dL_dcolor,
					   (glm::vec3*)dL_ddir,
					   (glm::vec3*)dL_dsh);
}

void BACKWARD::eval_sh(
	int P, int D, int M,
	const float* shs,
	const float* dirs,
	const bool* clamped,
	// input
	const float* dL_dcolor,
	// output
	float* dL_dsh,
	float* dL_ddir) {

	// printf("P=%d, D=%d, M=%d\n", P, D, M);
	eval_sh_CUDA<NUM_CHANNELS><<<(P + 255) / 256, 256>>>(
		P, D, M,
		shs,
		dirs,
		clamped,
		dL_dcolor,
		dL_dsh,
		dL_ddir);
	// sync
	cudaDeviceSynchronize();
}