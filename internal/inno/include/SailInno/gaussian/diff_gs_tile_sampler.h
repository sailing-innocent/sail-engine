#pragma once
/**
 * @file packages/gaussian/diff_gs_tile_sampler.h
 * @author sailing-innocent
 * @date 2024-03-07
 * @brief Tile Based Sampler for Discretely Sampling a list of standard Gaussian
 */

#include "SailInno/core/runtime.h"
#include "SailInno/helper/buffer_filler.h"

namespace sail::inno::gaussian {

class SAIL_INNO_API DiffGaussianTileSampler : public LuisaModule {
public:
	DiffGaussianTileSampler() = default;
	~DiffGaussianTileSampler() = default;

	void create(Device& device) noexcept;
	void create(Device& device, S<BufferFiller> buffer_filler) noexcept;

	void forward_impl(
		Device& device,
		Stream& stream,
		// params
		int num_gaussians,
		int height, int width,
		// input
		BufferView<float> means_2d,
		BufferView<float> covs_2d,
		BufferView<float> depth_features,
		BufferView<float> color_features,
		// output
		BufferView<float> target_img_buffer);

	void backward_impl(
		Device& device);

public:
	// component
	struct GeometryState {
		size_t scan_temp_storage_size;
		Buffer<int> scan_temp_storage;
		Buffer<float> means_2d_res;// 2 * P means 2d in resolution
		Buffer<int> radii;		   // P
		Buffer<float> covs_2d;	   // 3 * P
		Buffer<float> conic;	   // 3 * P
		Buffer<uint> tiles_touched;// P
		Buffer<uint> point_offsets;// P

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

	U<GeometryState> geom_state;
	U<TileState> tile_state;
	U<ImageState> img_state;

protected:
	void
	compile(Device& device) noexcept;
	// components
	uint2 m_blocks = {16u, 16u};
	uint2 m_grids = {1u, 1u};
	uint2 m_resolution;
	uint m_shared_mem_size = 256u;
	S<BufferFiller> mp_buffer_filler;
	int m_num_gaussians;

protected:
	// callable
	UCallable<void(float2, int, uint2&, uint2&, uint2, uint2)> mp_get_rect;

protected:
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
	U<Shader<2, uint2,	   // resolution
			 Buffer<float>,// target img
			 uint2,		   // grids
			 Buffer<uint>, // ranges
			 Buffer<uint>, // point_list
			 Buffer<float>,// means_2d
			 Buffer<float>,// features, P * 4
			 Buffer<float>,// conic
			 // save for backward
			 Buffer<uint>,// last_contributors
			 Buffer<float>// final_Ts
			 >>
		m_forward_render_shader;
};

}// namespace sail::inno::gaussian