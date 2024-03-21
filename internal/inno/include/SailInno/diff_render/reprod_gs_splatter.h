#pragma once
/**
 * @file packages/diff_render/gs/gaussian_splatter.h
 * @author sailing-innocent
 * @date 2023-12-27
 * @brief the gaussian splatter render
 */

#include "SailInno/core/runtime.h"
#include <luisa/runtime/buffer.h>
#include <luisa/runtime/stream.h>
#include "SailInno/helper/buffer_filler.h"
#include "SailInno/util/camera.h"

namespace sail::inno::render {

class SAIL_INNO_API ReprodGS : public LuisaModule {

public:
	ReprodGS() = default;
	~ReprodGS() = default;
	// API
	void create(Device& device) noexcept;

	virtual void forward_impl(
		// common params
		Device& device,
		Stream& stream,
		// output params
		int height, int width,
		// output
		BufferView<float> target_img_buffer,// hwc
		BufferView<int> radii,
		// params
		int num_gaussians, int sh_deg, int max_sh_deg,
		// input
		BufferView<float> xyz_buffer,
		BufferView<float> feature_buffer,// for color
		BufferView<float> opacity_buffer,
		BufferView<float> scale_buffer,
		BufferView<float> rotq_buffer,
		float scale_modifier,
		Camera& cam);

	void gaussian_proj_impl(
		Device& device,
		CommandList& cmdlist,
		int num_gaussians, int sh_deg, int max_sh_deg,
		float scale_modifier,
		BufferView<float> xyz_buffer,
		BufferView<float> feature_buffer,// for color
		BufferView<float> scale_buffer,
		BufferView<float> rotq_buffer,
		Camera& cam);

	virtual void backward_impl(
		Device& device,
		Stream& stream,
		// input
		BufferView<float> dL_d_pix,
		// output
		BufferView<float> dL_d_xyz,
		BufferView<float> dL_d_feature,
		BufferView<float> dL_d_opacity,
		BufferView<float> dL_d_scale,
		BufferView<float> dL_d_rotq,
		BufferView<float> dL_d_means_2d,// retained for trick
		// params
		BufferView<float> target_img_buffer,// hwc
		BufferView<float> xyz_buffer,
		BufferView<float> feature_buffer,// for color
		BufferView<float> opacity_buffer,
		BufferView<float> scale_buffer,
		BufferView<float> rotq_buffer);

	// component
	struct GeometryState {
		size_t scan_temp_storage_size;
		Buffer<int> scan_temp_storage;
		Buffer<float> means_2d;		   // 2 * P
		Buffer<float> means_2d_res;	   // 2 * P means 2d in resolution
		Buffer<float> depth_features;  // P
		Buffer<float> color_features;  // 3 * P
		Buffer<float> opacity_features;// P
		Buffer<float> covs_2d;		   // 3 * P
		Buffer<float> conic;		   // 3 * P
		Buffer<uint> tiles_touched;	   // P
		Buffer<uint> point_offsets;	   // P
		// method
		void allocate(Device& device, size_t size);
		void clear(Device& device, CommandList& cmdlist, BufferFiller& filler);
	};

	class TileState {
	public:
		size_t sort_temp_storage_size;
		Buffer<int> sort_temp_storage;
		Buffer<ulong> point_list_keys;
		Buffer<ulong> point_list_keys_unsorted;
		Buffer<uint> point_list_unsorted;
		Buffer<uint> point_list;

		// for shade
		Buffer<ulong> point_list_keys_shade;
		bool use_shade = false;

		void allocate(Device& device, size_t size);
		void clear(Device& device, CommandList& cmdlist, BufferFiller& filler);
	};
	class ImageState {
	public:
		Buffer<uint> ranges;// 2 pairs
		Buffer<uint> n_contrib;
		Buffer<float> accum_alpha;
		// method
		void allocate(Device& device, size_t size);
		void clear(Device& device, CommandList& cmdlist, BufferFiller& filler);
	};
	// componenet

	U<BufferFiller> mp_buffer_filler;

protected:
	// state
	U<GeometryState> geom_state;
	U<TileState> tile_state;
	U<ImageState> img_state;
	S<Camera> mp_camera;

	int m_num_gaussians;
	uint2 m_blocks = {16u, 16u};
	uint2 m_grids = {1u, 1u};
	uint m_shared_mem_size = 256u;
	uint2 m_resolution;
	int m_sh_deg;
	int m_max_sh_deg;
	float m_scale_modifier = 1.f;

	// compile shaders
	virtual void compile(Device& device) noexcept;
	virtual void compile_callables(Device& device) noexcept;
	virtual void compile_forward_preprocess_shader(Device& device) noexcept;
	virtual void compile_forward_render_shader(Device& device) noexcept;
	virtual void compile_backward_preprocess_shader(Device& device) noexcept;
	virtual void compile_backward_render_shader(Device& device) noexcept;
	virtual void compile_copy_with_keys_shader(Device& device) noexcept;
	virtual void compile_get_ranges_shader(Device& device) noexcept;

	// callables
	UCallable<float(float, uint)> mp_ndc2pix;
	UCallable<void(float2, int, uint2&, uint2&, uint2, uint2)> mp_get_rect;
	UCallable<float3x3(float3, float, float4)> mp_compute_cov_3d;
	UCallable<float3(float4, float4, float3x3, float4x4)> mp_compute_cov_2d;

	UCallable<float3(
		int, int, int,
		Buffer<float>,
		float3,
		Buffer<float>)>
		mp_compute_color_from_sh;

	UCallable<float3(
		int, int, int,
		Buffer<float>,
		float3,
		Buffer<float>,
		// input
		Buffer<float>,// dL_d_color_feature
		// output
		Buffer<float>// dL_d_feat
		)>
		mp_compute_color_from_sh_backward;

	// shaders
	U<Shader<1, int,// num_gaussians
			 uint2, // resolution
			 uint2, // grids
			 // input
			 Buffer<float>,// screen means
			 Buffer<float>,// screen cov2d
			 // output
			 Buffer<float>,// conic
			 Buffer<float>,// means_2d_screen
			 Buffer<uint>, // tiles_touched
			 Buffer<int>   // radii
			 >>
		m_forward_tile_split_shader;

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
			 Buffer<float>,// color_features // 4 * P
			 Buffer<float>,// covs_2d // 3 * P
			 // PARAMS
			 float3,  // cam_pos
			 float4,  // focal_x, focal_y, tan_fov_x, tan_fov_y
			 float4x4,// view_matrix
			 float4x4 // proj_matrix
			 >>
		m_forward_preprocess_shader;

	U<Shader<1,
			 // input
			 Buffer<float>,// dL_d_means_2d
			 Buffer<float>,// dL_d_conic
			 Buffer<float>,// dL_d_color_feature
			 // output
			 Buffer<float>,// dL_d_xyz
			 Buffer<float>,// dL_d_feat
			 Buffer<float>,// dL_d_scale
			 Buffer<float>,// dL_d_rotq
			 // params
			 int, int, int,// P, D, M
			 uint2, uint2, // resolution, grids
			 Buffer<float>,// means_3d
			 Buffer<float>,// shs
			 Buffer<float>,// scale_buffer
			 Buffer<float>,// qvec_buffer
			 Buffer<float>,// opacity_features
			 Buffer<float>,// color_feature
			 Buffer<float>,// conics
			 // camera
			 float3, // cam_pos
			 float4, // focal_x, focal_y, tan_fov_x, tan_fov_y
			 float4x4// view_matrix
			 >>
		m_backward_preprocess_shader;

	U<Shader<1, int,	   // P
			 Buffer<float>,// points_xy
			 Buffer<uint>, // offsets
			 Buffer<int>,  // radii
			 Buffer<float>,// depth_features
			 Buffer<ulong>,// keys_unsorted
			 Buffer<uint>, // values_unsorted
			 uint2, uint2  // blocks & grids
			 >>
		m_copy_with_keys_shader;

	U<Shader<1, int,	   // num_rendered
			 Buffer<ulong>,// point_list_keys
			 Buffer<uint>  // ranges
			 >>
		m_get_ranges_shader;

	U<Shader<2,
			 uint2,		   // resolution
			 Buffer<float>,// target img
			 // input
			 uint2,		   // grids
			 Buffer<uint>, // ranges
			 Buffer<uint>, // point_list
			 Buffer<float>,// means_2d, P x 2
			 Buffer<float>,// conic, P x 3
			 Buffer<float>,// opacity_features, P
			 Buffer<float>,// color_features, P * 3
			 // save for backward
			 Buffer<uint>,// last_contributors
			 Buffer<float>// final_Ts
			 >>
		m_forward_render_shader;

	U<Shader<2,
			 // input
			 Buffer<float>,// dL_d_pix
			 // output
			 Buffer<float>,// dL_d_means_2d
			 Buffer<float>,// dL_d_conic
			 Buffer<float>,// dL_d_color_feature
			 Buffer<float>,// dL_d_opacity
			 // params
			 uint2,		   // resolution
			 uint2,		   // grids
			 Buffer<float>,// result_img
			 Buffer<uint>, // ranges
			 Buffer<uint>, // point_list
			 Buffer<float>,// means_2d
			 Buffer<float>,// conic
			 Buffer<float>,// opacity
			 Buffer<float>,// features
			 Buffer<uint>, // final_contrib
			 Buffer<float> // final_Ts
			 >>
		m_backward_render_shader;
};

}// namespace sail::inno::render