#pragma once

/**
 * @file packages/painter/painter3d.h
 * @author sailing-innocent
 * @date 2023-12-27
 * @brief The 3D Painter Base class, Pure Color
 */

#include "painter_base.h"
#include "SailInno/util/camera.h"

namespace sail::inno {

class SAIL_INNO_API Painter3D : public PainterBase {

public:
	Painter3D() = default;
	virtual ~Painter3D() = default;
	// virtual void create(Device& device) noexcept override;
	// virtual void paint(CommandList& cmdlist, ImageView<float> out_img, int w, int h) noexcept override;
	virtual void update() noexcept;
	void update_camera(Camera& camera) noexcept;// update camera binding

protected:
	S<Camera> mp_camera;
	luisa::float4x4 m_view_matrix;
	luisa::float4x4 m_proj_matrix;
	luisa::float3 m_cam_pos = {0.f, 0.f, 0.f};
};

}// namespace sail::inno