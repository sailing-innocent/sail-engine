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
	int num_task;
	int num_thread_up;
};

}// namespace sail::inno::sph

LUISA_STRUCT(sail::inno::sph::NeighborState, middle, bt_offset, num_task, num_thread_up){};

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
	Neighbor(SPHSolver& solver) noexcept;
	class TaskState {
		friend class Neighbor;

	public:
		Buffer<int> particle_offset;// offset in cell
		Buffer<int> particle_count;
		Buffer<int> cell_index;

	private:
		void allocate(Device& device, size_t size);
	};

	class CellState {
		friend class Neighbor;

	public:
		Buffer<int> particle_offset;
		Buffer<int> particle_count;
		Buffer<int> particle_offset_hash;// size << 1
		Buffer<int> particle_count_hash; // size << 1
		Buffer<int> task_count;
		Buffer<int> task_offset;

	private:
		void allocate(Device& device, size_t size);
	};
	NeighborState& state() noexcept { return m_state; }
	TaskState& task_state() noexcept { return *mp_task_state; }
	CellState& cell_state() noexcept { return *mp_cell_state; }
	// temp scan storage
	Buffer<int> m_temp_storage;// for scan
	// resources
	Buffer<int> m_hash;
	Buffer<int> m_cell_index;
	Buffer<int> m_index;
	Buffer<int> m_thread_index;
	Buffer<int> m_thread_to_data_index;
	Buffer<int> tmp_id;
	Buffer<float3> tmp_pos;
	Buffer<float3> tmp_vel;

	// parameters
	size_t m_num_grids = 0;
	size_t m_num_cells = 0;
	size_t m_num_tasks = 0;
	size_t m_num_hashs = 0;
	float m_cell_size = 0.f;
	int m_threads_upper_limit = 0;

	size_t size() const noexcept { return m_size; }
	void solve(Device& device, CommandList& cmdlist) noexcept;

private:
	// life cycle
	// only friend class can call these methods
	void create(Device& device) noexcept;
	void compile(Device& device) noexcept;
	void reset() noexcept;
	void allocate(Device& device, size_t size) noexcept;

	// util
	int get_thread_up(int x) noexcept;

	// state
	NeighborState m_state;// for both side
	U<TaskState> mp_task_state;
	U<CellState> mp_cell_state;

	// params
	size_t m_size = 0;	  // particle size
	size_t m_capacity = 0;// influence the tasks allocated

	// shaders
	U<Shader<1, int, Buffer<int>>> ms_clear_cell;
	U<Shader<1, int, int, float, Buffer<float3>>> ms_count_sort_cell_sum;
	// copy from tmp
	// count sort result
	// cal block
	// count sort cell sum 2
	// count sort result 2
	// find middle value
	// arrange task
};

}// namespace sail::inno::sph