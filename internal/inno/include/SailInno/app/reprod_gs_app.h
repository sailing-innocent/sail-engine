#pragma once

/**
 * @file app/diff_render/gaussian_splatter_app.h
 * @author sailing-innocent
 * @date 2023-12-27
 * @brief the gaussian splatter app
 */

#include "SailInno/app/base.h"
#include "SailInno/diff_render/reprod_gs_splatter.h"
#include <array>

namespace sail::inno::app {

class SAIL_INNO_API ReprodGSApp : public BaseApp {
public:
	ReprodGSApp() = default;
	virtual ~ReprodGSApp() = default;
	void create(luisa::string& cwd, luisa::string& device_name) override;

	void forward(
		int height, int width,
		int64_t target_img,
		int64_t radii,
		int P, int sh_deg, int max_sh_deg,
		int64_t xyz, int64_t feat, int64_t opacity, int64_t scales, int64_t rotqs, float scale_modifier,
		std::array<float, 3> cam_pos, float fov_rad, std::array<float, 16> view_matrix_arr, std::array<float, 16> proj_matrix_arr);

	void backward(
		// input
		int64_t dL_d_pix,
		// output
		int64_t dL_d_xyz,
		int64_t dL_d_feature,
		int64_t dL_d_opacity,
		int64_t dL_d_scale,
		int64_t dL_d_rotq,
		int64_t dL_d_means_2d,
		// params
		int64_t target_img_buffer,// hwc
		int64_t xyz_buffer,
		int64_t feature_buffer,// for color
		int64_t opacity_buffer,
		int64_t scale_buffer,
		int64_t rotq_buffer);

protected:
	U<render::ReprodGS> mp_render;
	// save for backward
	int m_width;
	int m_height;
	int m_P;
	int m_sh_deg;
	int m_max_sh_deg;
};

}// namespace sail::inno::app