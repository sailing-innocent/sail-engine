#pragma once

/**
 * @file source/package/solver/fluid/sph/neighbor.h
 * @author sailing-innocent
 * @date 2023-02-23
 * @brief The Neighbor Search for SPH
 */

#include "SailInno/core/runtime.h"
#include "sph_executor.h"
#include <luisa/dsl/sugar.h>

namespace sail::inno::sph {

struct NeighborState {
	int middle;
	int bt_offset;
	int num_tasks;
	int num_threads;
};

}// namespace sail::inno::sph

LUISA_STRUCT(sail::inno::sph::NeighborState, middle, bt_offset, num_tasks, num_threads){};

namespace sail::inno::sph {

class SPHSolver;

class SAIL_INNO_API Neighbor : public SPHExecutor {
	friend class SPHSolver;
	template<size_t I, typename... Ts>
	using Shader = luisa::compute::Shader<I, Ts...>;
	using Int = luisa::compute::Int;
	using Float = luisa::compute::Float;

public:
	// ctor & dtor
	class TaskState {
		friend class Neighbor;
	};
	class CellState {
		friend class Neighbor;
	};

	Neighbor(SPHSolver& solver) noexcept;
	TaskState& task_state() noexcept { return *mp_task_state; }
	CellState& cell_state() noexcept { return *mp_cell_state; }

public:
	// parameters
	size_t m_num_grids = 0;
	size_t m_num_cells = 0;
	size_t m_num_tasks = 0;
	size_t m_num_hashs = 0;
	float m_cell_size = 0.f;
	int m_threads_upper_limit = 0;

public:
	size_t size() const noexcept { return m_size; }
	void solve(CommandList& cmdlist) noexcept;

private:
	// life cycle
	// only friend class can call these methods
	void create(Device& device) noexcept;
	void compile(Device& device) noexcept;
	void reset() noexcept;
	void allocate(Device& device, size_t size) noexcept;
	void after_solve() noexcept;

private:
	// state
	NeighborState m_state;
	U<TaskState> mp_task_state;
	U<CellState> mp_cell_state;

	size_t m_size;
	size_t m_max_size;

private:
	// resources

private:
	// shaders
	U<Shader<1, int, Buffer<int>>> ms_clear_cell;
	U<Shader<1, int, int, float, Buffer<float3>>> ms_count_sort_cell_sum;
};

}// namespace sail::inno::sph