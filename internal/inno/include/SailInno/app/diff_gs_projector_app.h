#pragma once

/**
 * @file app/GS/diff_GS_projector_app.h
 * @author sailing-innocent
 * @date 2024-03-07
 * @brief The Diff GS Projector
 */

#include "SailInno/app/base.h"
#include "SailInno/gaussian/diff_gs_projector.h"

#include <array>

namespace sail::inno::app {

class SAIL_INNO_API DiffGSProjectorApp : public BaseApp {
public:
	DiffGSProjectorApp() = default;
	virtual ~DiffGSProjectorApp() = default;
	virtual void create(luisa::string& cwd, luisa::string& device_name) override;

	void sync() noexcept;

	void forward(
		int num_gaussians, int sh_deg, int max_sh_deg, float scale_modifier,
		// input
		int64_t xyz,
		int64_t feature,
		int64_t scale,
		int64_t rotq,
		// output
		int64_t means_2d,
		int64_t covs_2d,
		int64_t depth_features,
		int64_t color_features,
		// camera
		std::array<float, 3> cam_pos, std::array<float, 16> view_matrix);
	void backward();

protected:
	U<gaussian::DiffGaussianProjector> mp_projector;

	// save for backward
	// int   m_num_GSs;
	// int   m_sh_deg;
	// int   m_max_sh_deg;
	// float m_scale_modifier;
	// int   m_resw;
	// int   m_resh;
	// int   m_cam_pos;
	// float m_fov_rad;
	// int   m_view_matrix;
	// int   m_proj_matrix;
	// int   m_means_2d;
	// int   m_depth_features;
	// int   m_color_features;
	// int   m_covs_2d;
	// int   m_xyz;
	// int   m_feature;
	// int   m_opacity;
	// int   m_scale;
	// int   m_rotq;
};

}// namespace sail::inno::app
