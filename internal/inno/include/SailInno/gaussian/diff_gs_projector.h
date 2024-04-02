#pragma once
/**
 * @file packages/gaussian/diff_gs_projector/projector.h
 * @author sailing-innocent
 * @date 2024-03-07
 * @brief Projecting a list of 3D Gaussians into 2D Gaussian List, Differentiable
 */

#include "SailInno/core/runtime.h"
#include "SailInno/helper/buffer_filler.h"
#include "SailInno/util/camera.h"

#include <luisa/runtime/buffer.h>
#include <luisa/runtime/stream.h>

namespace sail::inno::gaussian {

// @forward
// input: xyz, feature, opacity, scale, rotq in world space
// output: xy, conic, color, opacity in screen space

class DiffGaussianProjector : public LuisaModule {
public:
	DiffGaussianProjector() = default;
	// share buffer filler

	~DiffGaussianProjector() = default;
	// API
	void create(Device& device) noexcept;
	void create(Device& deivce, S<BufferFiller> p_buffer_filler) noexcept;

	void forward_impl(
		Device& device,
		CommandList& cmdlist,
		int num_gaussians,
		int sh_deg,
		int max_sh_deg,
		float scale_modifier,
		// input // scene
		BufferView<float> xyz_buffer,
		BufferView<float> feature_buffer,// for color
		BufferView<float> scale_buffer,
		BufferView<float> rotq_buffer,
		// input // camera
		Camera& cam,
		// output // screen space
		BufferView<float> means_2d,
		BufferView<float> depth_features,
		BufferView<float> color_features,
		BufferView<float> covs_2d);

	void backward_impl(
		Device& device);

	// components
	S<BufferFiller> mp_buffer_filler;
	uint2 m_blocks = {16u, 16u};

private:
	void compile(Device& device) noexcept;

protected:
	// callables

	UCallable<float3x3(float3, float, float4)> mp_compute_cov_3d;
	UCallable<float3(float3, float3x3, float4x4)> mp_compute_cov_2d;
	UCallable<float3(
		int, int, int,
		Buffer<float>,
		float3,
		Buffer<float>)>
		mp_compute_color_from_sh;
	// shader
	U<Shader<1, int, int, int,// P, D, M
			 Buffer<float>,	  // means_3d
			 Buffer<float>,	  // feat_buffer
			 Buffer<float>,	  // scale_buffer
			 Buffer<float>,	  // rotq_buffer
			 // params
			 float,// scale_modifier
			 // output
			 Buffer<float>,// means_2d // 2 * P
			 Buffer<float>,// depth_features // P
			 Buffer<float>,// color_features // 3 * P
			 Buffer<float>,// covs_2d // 3 * P
			 // PARAMS
			 float3, // cam_pos
			 float4x4// view_matrix
			 >>
		m_forward_preprocess_shader;
};

}// namespace sail::inno::gaussian