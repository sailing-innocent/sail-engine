/**
 * @file solver/sph/neighbor/neighbor_impl.cpp
 * @author sailing-innocent
 * @date 2023-03-25
 * @brief The Neighbor Search for SPH impl
 */

#include "SailInno/solver/sph/neighbor.h"
#include "SailInno/solver/sph/solver.h"
#include "SailInno/solver/sph/sph_executor.h"
#include "SailInno/solver/sph/fluid_particles.h"

namespace sail::inno::sph {
void Neighbor::TaskState::allocate(Device& device, size_t size) {
	particle_offset = device.create_buffer<int>(size);
	particle_count = device.create_buffer<int>(size);
	cell_index = device.create_buffer<int>(size);
}

void Neighbor::CellState::allocate(Device& device, size_t size) {
	particle_offset = device.create_buffer<int>(size);
	particle_count = device.create_buffer<int>(size);
	particle_offset_hash = device.create_buffer<int>(size << 1);
	particle_count_hash = device.create_buffer<int>(size << 1);
	task_count = device.create_buffer<int>(size);
	task_offset = device.create_buffer<int>(size);
}

}// namespace sail::inno::sph

namespace sail::inno::sph {

Neighbor::Neighbor(SPHSolver& solver) noexcept : SPHExecutor{solver} {
	// init its state component
	mp_task_state = luisa::make_unique<TaskState>();
	mp_cell_state = luisa::make_unique<CellState>();
}

void Neighbor::allocate(Device& device, size_t size) noexcept {
	m_hash = device.create_buffer<int>(size);
	m_cell_index = device.create_buffer<int>(size);
	m_index = device.create_buffer<int>(size);
	m_thread_index = device.create_buffer<int>(size);
	m_thread_to_data_index = device.create_buffer<int>(size);

	tmp_id = device.create_buffer<int>(size);
	tmp_pos = device.create_buffer<float3>(size);
	tmp_vel = device.create_buffer<float3>(size);
}

void Neighbor::reset() noexcept {
	m_size = solver().particles().size();
	// set cell size the kernel radius h
	m_cell_size = solver().param().kernel_radius;
	// cell size should not be smaller than min_cell_size
	m_cell_size = std::max(m_cell_size, solver().config().min_cell_size);
	// allocate the grid
	m_num_grids = (solver().config().world_size + m_cell_size - 1) / m_cell_size;

	m_num_cells = m_num_grids * m_num_grids * m_num_grids;

	// allocate tasks
	m_num_tasks = (m_capacity + 32 - 1) / 32 + m_num_cells;
	m_num_hashs = m_num_cells * 2;

	m_state.num_thread_up = get_thread_up(m_size);// upper limit
}

void Neighbor::create(Device& device) noexcept {
	m_capacity = solver().config().n_capacity;
	// set initial data
	reset();
	// allocate the maximum value
	int max_num_grids = (solver().config().world_size / solver().config().min_cell_size) + 1;
	int max_num_cells = max_num_grids * max_num_grids * max_num_grids;
	int max_num_tasks = m_capacity / 32 + max_num_cells;
	size_t num_cells = max_num_cells;
	size_t num_tasks = max_num_tasks;

	// allocate memory
	allocate(device, m_capacity);
	mp_task_state->allocate(device, num_tasks);
	mp_cell_state->allocate(device, num_cells);
	// get temp storage size
	size_t temp_storage_size = -1;
	solver().device_parallel().scan_exclusive_sum(temp_storage_size,
												  mp_cell_state->particle_count_hash,
												  mp_cell_state->particle_offset_hash,
												  0,
												  num_cells << 1);
	// allocate temp_storage
	m_temp_storage = device.create_buffer<int>(temp_storage_size);
}

int Neighbor::get_thread_up(int x) noexcept {
	int block_int = int(solver().config().n_blocks);
	return (x / block_int + 2) * block_int;
}

void Neighbor::solve(CommandList& cmdlist) noexcept {
	// core solve algorithm
	int num_cells = m_num_cells;
	int num_particles = m_size;
	int n_grids = m_num_grids;
	int n_hashs = m_num_hashs;
	float cell_size = m_cell_size;
	auto& particles = solver().particles();

	// count particles in each cell
	// scan to get the offset
	// count task in each cell
	// scan to get the offset
	// count hash for each partilce
	// scan to get the hash offset
	// find [Sparse|Dence] middel value
	// arrange tasks
	// sync

	// cmdlist << solver().filler().fill(m_cell->particle_count, 0)
	//         << solver().filler().fill(m_cell->task_count, 0)
	//         << (*count_sort_cell_sum)(num_particles, n_grids, cell_size, particles.m_pos).dispatch(num_particles);
	// solver().device_parallel().scan_exclusive_sum(cmdlist, m_scan_temp->temp_storage, m_cell->particle_count, m_cell->particle_offset, num_cells);

	// cmdlist << (*count_sort_result)(num_particles, n_grids, cell_size, particles.m_id, particles.m_pos, particles.m_vel).dispatch(num_particles)

	//         // Count tasks in each cell
	//         << (*cal_block)(num_cells).dispatch(num_cells)
	//         << (*copy_from_tmp)(num_particles, particles.m_id, particles.m_pos, particles.m_vel).dispatch(num_particles);
	// solver().device_parallel().scan_exclusive_sum(cmdlist, m_scan_temp->temp_storage, m_cell->task_count, m_cell->task_offset, num_cells);

	// // Count hash for each particle
	// cmdlist << solver().filler().fill(m_cell->particle_count_hash, 0)
	//         << (*count_sort_cell_sum_2)(num_particles, n_grids, num_cells, cell_size, particles.m_pos).dispatch(num_particles);
	// solver().device_parallel().scan_exclusive_sum(cmdlist, m_scan_temp->temp_storage, m_cell->particle_count_hash, m_cell->particle_offset_hash, num_cells << 1);

	// cmdlist << (*count_sort_result_2)(num_particles).dispatch(num_particles)

	//         // Find dividing line between sparse and dense particles
	//         // [Sparse, Dense]
	//         << (*find_middle_value)(num_particles, num_particles, num_cells, n_hashs).dispatch(num_particles)
	//         // Arrange particles for each task
	//         << (*arrange_task)(num_cells, num_cells).dispatch(num_cells)
	//         << m_data_gpu.copy_to(m_data_cpu.data());
}

}// namespace sail::inno::sph
