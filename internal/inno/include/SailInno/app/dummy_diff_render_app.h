#pragma once

/**
 * @file app/diff_render/dummy_diff_render_app.h
 * @author sailing-innocent
 * @date 2023-12-29
 * @brief the gaussian splatter app
 */

#include "SailInno/app/base.h"
#include "SailInno/diff_render/dummy_diff_render.h"

namespace sail::inno::app {

class SAIL_INNO_API DummyDiffRenderApp : public BaseApp {
public:
	DummyDiffRenderApp() = default;
	virtual ~DummyDiffRenderApp() = default;
	void create(luisa::string& cwd, luisa::string& device_name) override;
	void forward(int height, int width, int64_t source_img, int64_t target_img) noexcept;
	void backward(int height, int width, int64_t dL_dtpix, int64_t dL_dspix) noexcept;

protected:
	U<render::DummyDiffRender> mp_render;
};

}// namespace sail::inno::app