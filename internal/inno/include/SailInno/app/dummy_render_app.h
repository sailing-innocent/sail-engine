#pragma once

/**
 * @file app/dummy_render_app.h
 * @author sailing-innocent
 * @date 2023-11-16
 * @brief the dummy renderer app
 */

#include "SailInno/app/base.h"
#include "SailInno/render/dummy_render.h"

namespace sail::inno::app {

class SAIL_INNO_API DummyRenderApp : public BaseApp {
public:
	DummyRenderApp() = default;
	virtual ~DummyRenderApp() = default;
	virtual void create(luisa::string& cwd, luisa::string& device_name) override;
	luisa::vector<float> render_cpu(int height, int width);
	void render_cuda(int height, int width, int64_t target_img);

protected:
	U<sail::inno::render::DummyRender> mp_render;
};

}// namespace sail::inno::app