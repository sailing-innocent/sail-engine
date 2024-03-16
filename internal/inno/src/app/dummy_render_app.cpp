#include "SailInno/app/dummy_render_app.h"
#include <luisa/core/logging.h>
#include <luisa/runtime/buffer.h>

namespace sail::inno::app {

void DummyRenderApp::create(luisa::string& cwd, luisa::string& device_name) {
	LUISA_INFO("DummyRenderApp::create");
	luisa::compute::Context context{cwd.c_str()};
	mp_device = luisa::make_shared<luisa::compute::Device>(context.create_device(device_name.c_str()));
	mp_stream = luisa::make_shared<luisa::compute::Stream>(mp_device->create_stream());
	mp_render = luisa::make_unique<render::DummyRender>();
	mp_render->compile(*mp_device);
}

luisa::vector<float> DummyRenderApp::render_cpu(int height, int width) {
	LUISA_INFO("DummyRenderApp::render_cpu");
	auto res = luisa::vector<float>(height * width * 3);
	for (auto i = 0; i < height * width; i++) {
		res[3 * i + 0] = 1.0f;
		res[3 * i + 1] = 0.0f;
		res[3 * i + 2] = 0.0f;
	}
	return res;
}

void DummyRenderApp::render_cuda(int height, int width, int64_t target_img) {
	LUISA_INFO("DummyRenderApp::render_cuda");
	luisa::compute::Buffer<float> target_img_buf = mp_device->import_external_buffer<float>((void*)target_img, height * width * 3);
	luisa::compute::CommandList cmdlist;
	mp_render->render(cmdlist, target_img_buf, height, width);
	(*mp_stream) << cmdlist.commit() << luisa::compute::synchronize();
}

}// namespace sail::inno::app