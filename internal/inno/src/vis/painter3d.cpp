/**
 * @file packages/painter/painter3d.cpp
 * @author sailing-innocent
 * @date 2023/12/27
 * @brief The 3D Painter Base class, Impl
*/

#include "SailInno/vis/painter3d.h"

namespace sail::inno {

void Painter3D::update_camera(Camera& camera) noexcept {
	mp_camera = luisa::make_shared<Camera>(camera);
	update();
};
void Painter3D::update() noexcept {
	// update the matrix according to camera
	m_view_matrix = mp_camera->view_matrix();
	m_proj_matrix = mp_camera->proj_matrix();
	m_cam_pos = mp_camera->pos();
	// update scene
}

}// namespace sail::inno