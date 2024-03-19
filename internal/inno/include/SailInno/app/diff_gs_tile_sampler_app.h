#pragma once

/**
 * @file app/GS/diff_gs_tile_sampler_app.h
 * @author sailing-innocent
 * @date 2024-03-08
 * @brief The Diff GS Tile Sampler
 */

#include "SailInno/app/base.h"
#include "SailInno/gaussian/diff_gs_tile_sampler.h"

namespace sail::inno::app {

class SAIL_INNO_API DiffGSTileSamplerApp : public BaseApp {
public:
	DiffGSTileSamplerApp() = default;
	virtual ~DiffGSTileSamplerApp() = default;
	void create(luisa::string& cwd, luisa::string& device_name) override;

	void forward(// params
		int num_gaussians, int height, int width, float fov_rad,
		// input
		int64_t means_2d, int64_t covs_2d, int64_t depth_features, int64_t opacity_features, int64_t color_features,
		// output
		int64_t target_img_buffer);
	void backward(
		// input
		int64_t dL_dpix,
		// params
		int64_t covs_2d,
		int64_t opacity_features,
		int64_t color_features,
		// output
		int64_t dL_d_means_2d,
		int64_t dL_d_covs_2d,
		int64_t dL_d_opacity_features,
		int64_t dL_d_color_features);

protected:
	int m_num_gaussians;
	int m_height;
	int m_width;
	U<gaussian::DiffGaussianTileSampler> mp_sampler;
};

}// namespace sail::inno::app
