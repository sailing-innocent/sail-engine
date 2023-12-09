#pragma once

/**
 * @file packages/painter/point_painter.h
 * @author sailing-innocent
 * @date 2023/12/27
 * @brief 3D Point Painter
 */

#include "SailInno/vis/painter3d.h"
#include "SailInno/render/point_render.h"

namespace sail::inno {

class SAIL_INNO_API PointPainter : public Painter3D {
	using Device = luisa::compute::Device;
	using CommandList = luisa::compute::CommandList;
	template<typename T>
	using BufferView = luisa::compute::BufferView<T>;
	template<typename T>
	using ImageView = luisa::compute::ImageView<T>;

public:
	PointPainter() = default;
	int m_stride = 3;
	virtual ~PointPainter() = default;
	virtual void create(Device& device) noexcept override;
	virtual void paint(CommandList& cmdlist, ImageView<float> out_img, int w, int h) noexcept override;
	void update_point(int point_num, BufferView<float> point_xyz, BufferView<float> point_color) noexcept;

protected:
	int m_point_num;
	BufferView<float> m_point_xyz;
	BufferView<float> m_point_color;
	S<render::PointRender> mp_point_render;
};

}// namespace sail::inno
