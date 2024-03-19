#pragma once

/**
 * @file app/render/point_render_app
 * @author sailing-innocent
 * @date 2023-12-27
 * @brief the pointlist scene render app
*/

#include "SailInno/app/base.h"
#include "SailInno/render/point_render.h"
#include <array>

namespace sail::inno::app {

class SAIL_INNO_API PointRenderApp : public BaseApp {
public:
	PointRenderApp() = default;
	virtual ~PointRenderApp() = default;
	virtual void create(luisa::string& cwd, luisa::string& device_name) override;
	void render_cuda(int height, int width, int64_t target_img, int P, int64_t xyz, int64_t color, std::array<float, 16> view_matrix_arr, std::array<float, 16> proj_matrix_arr);

protected:
	U<render::PointRender> mp_render;
};

}// namespace sail::inno::app
