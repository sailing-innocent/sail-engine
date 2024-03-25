/**
 * @file app/point_render_app.cpp
 * @author sailing-innocent
 * @date 2023-12-27
 * @brief the pointlist scene render app impl
*/

#include <luisa/runtime/buffer.h>
#include "SailInno/app/point_render_app.h"
#include "SailInno/util/misc/mat_helper.h"

using namespace luisa;
using namespace luisa::compute;

namespace sail::inno::app {

void PointRenderApp::create(luisa::string& cwd, luisa::string& device_name) {
	LUISA_INFO("PointRenderApp::create");
	Context context{cwd.c_str()};
	mp_device = luisa::make_shared<Device>(context.create_device(device_name.c_str()));
	mp_stream = luisa::make_shared<Stream>(mp_device->create_stream());
	mp_render = luisa::make_unique<render::PointRender>();
	mp_render->compile(*mp_device);
}

void PointRenderApp::render_cuda(int height, int width, int64_t target_img, int P, int64_t xyz, int64_t color, std::array<float, 16> view_matrix_arr, std::array<float, 16> proj_matrix_arr) {
	CommandList cmdlist;
	Buffer<float> xyz_buf = mp_device->import_external_buffer<float>((void*)xyz, P * 3);
	Buffer<float> color_buf = mp_device->import_external_buffer<float>((void*)color, P * 3);
	Buffer<float> target_img_buf = mp_device->import_external_buffer<float>((void*)target_img, width * height * 3);

	auto view_matrix = arr16_mat44(view_matrix_arr);
	auto proj_matrix = arr16_mat44(proj_matrix_arr);

	mp_render->render(cmdlist, target_img_buf, width, height, P, xyz_buf.view(), color_buf.view(), view_matrix, proj_matrix);

	(*mp_stream) << cmdlist.commit() << synchronize();
}

}// namespace sail::inno::app