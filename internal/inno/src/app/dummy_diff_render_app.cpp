/**
 * @file app/diff_render/gaussian_splatter_app.cpp
 * @author sailing-innocent
 * @date 2023-12-29
 * @brief the gaussian scene render app impl
 */
#include "SailInno/app/dummy_diff_render_app.h"
#include <luisa/runtime/buffer.h>
using namespace luisa;
using namespace luisa::compute;

namespace sail::inno::app {

void DummyDiffRenderApp::create(luisa::string& cwd, luisa::string& device_name) {
	LUISA_INFO("DummyDiffRenderApp::create");
	Context context{cwd.c_str()};
	mp_device = luisa::make_shared<Device>(context.create_device(device_name.c_str()));
	mp_stream = luisa::make_shared<Stream>(mp_device->create_stream());
	mp_render = luisa::make_unique<render::DummyDiffRender>();
	mp_render->create(*mp_device);
}

void DummyDiffRenderApp::forward(int height, int width, int64_t source_img, int64_t target_img) noexcept {
	Buffer<float> source_img_buf = mp_device->import_external_buffer<float>((void*)source_img, width * height * 3);
	Buffer<float> target_img_buf = mp_device->import_external_buffer<float>((void*)target_img, width * height * 3);
	CommandList cmdlist;
	mp_render->forward_impl(
		cmdlist,
		width, height,
		source_img_buf,
		target_img_buf);
	(*mp_stream) << cmdlist.commit() << synchronize();
}

void DummyDiffRenderApp::backward(int height, int width, int64_t dL_dtpix, int64_t dL_dspix) noexcept {
	Buffer<float> dL_dtpix_buf = mp_device->import_external_buffer<float>((void*)dL_dtpix, width * height * 3);
	Buffer<float> dL_dspix_buf = mp_device->import_external_buffer<float>((void*)dL_dspix, width * height * 3);
	CommandList cmdlist;
	mp_render->backward_impl(
		cmdlist,
		width, height,
		dL_dtpix_buf,
		dL_dspix_buf);
	(*mp_stream) << cmdlist.commit() << synchronize();
}

}// namespace sail::inno::app