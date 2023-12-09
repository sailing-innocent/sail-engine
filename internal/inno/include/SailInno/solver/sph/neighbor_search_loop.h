#pragma once
/**
 * @file source/package/solver/fluid/sph/neighbor_search_loop.h
 * @author sailing-innocent
 * @date 2023-02-23
 * @brief The Neighbor Search Interface
 */

#include <luisa/dsl/sugar.h>
#include "neighbor.h"

namespace sail::inno::sph {

using SMEM_int = luisa::compute::Shared<int>;
using SMEM_int_ptr = luisa::shared_ptr<luisa::compute::Shared<int>>;
using SMEM_float = luisa::compute::Shared<float>;
using SMEM_float_ptr = luisa::shared_ptr<luisa::compute::Shared<float>>;
using SMEM_float3 = luisa::compute::Shared<luisa::float3>;
using SMEM_float3_ptr = luisa::shared_ptr<luisa::compute::Shared<luisa::float3>>;

class Neighbor;
class SPHFluidParticles;

template<typename BeginF, typename NewSmemF, typename LoadSmemF, typename LoopF, typename EndF>
inline void task_search(
	Neighbor& neighbor,
	SPHFluidParticles& particles,
	const luisa::compute::Int n_grids,
	const luisa::compute::Int n_threads,
	const luisa::compute::Int n_cta,
	const luisa::compute::Float cell_size,
	BeginF&& f_read,
	NewSmemF&& f_new_smem,
	LoadSmemF&& f_load_smem,
	LoopF&& f_loop,
	EndF&& f_write) {
	using namespace luisa;
	using namespace luisa::compute;

	auto& task_state = neighbor.task_state();
	auto& cell_state = neighbor.cell_state();

	auto t = dispatch_id().x;	  // global thread index
	auto block_x = block_id().x;  // block index
	auto thread_x = thread_id().x;// thread index in block
}

}// namespace sail::inno::sph