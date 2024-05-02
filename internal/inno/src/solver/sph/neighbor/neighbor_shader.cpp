/**
 * @file solver/sph/neighbor/neighbor_shader.cpp
 * @author sailing-innocent
 * @date 2023-03-25
 * @brief The Neighbor Search for SPH impl Shader 
 */

#include "SailInno/solver/sph/neighbor.h"
#include <luisa/dsl/sugar.h>
#include "SailInno/helper/grid_stride_loop.h"
#include "SailInno/solver/sph/solver.h"

namespace sail::inno::sph {

void Neighbor::compile(Device& device) noexcept {
	using namespace luisa;
	using namespace luisa::compute;

	const auto dim = 3;
	const auto n_blocks = solver().config().n_blocks;
	const int n_blocks_int = int(n_blocks);
	const auto n_threads = solver().config().n_threads;

	// Callable
	Callable ijk_to_cell_index = [&](UInt3 coord, Int n_grids) noexcept {
		using T = vector_expr_element_t<decltype(coord)>;
		auto p = clamp(coord, 0u, luisa::compute::UInt(n_grids - 1));
		return p.x + p.y * n_grids + p.z * n_grids * n_grids;
	};

	Callable pos_to_cell_index = [&](Float3 pos, Int n_grids, Float cell_size) noexcept {
		UInt3 coord;
		for (auto i = 0; i < dim; i++) {
			coord[i] = UInt(pos[i] / cell_size);
		}
		return ijk_to_cell_index(coord, n_grids);
	};

	// auto ceil_int = [&](auto a, auto b) noexcept { return (a + b - 1) / b; };
	// auto floor_int = [&](auto a, auto b) noexcept { return a / b; };

	// Callable cell_index_to_ijk = [&](UInt index, Int n_grids) noexcept {
	//     Int z = Int(index / (n_grids * n_grids));
	//     index = index % (n_grids * n_grids);
	//     Int y = Int(index / (n_grids));
	//     index = index % (n_grids);
	//     Int x = Int(index);
	//     return make_int3(x, y, z);
	// };

	// Callable cell_pos_to_cell_index = [&](UInt3 cell_pos, Int n_grids) noexcept {
	//     Int res = def(INVALID_CELL_INDEX);
	//     $if(cell_pos.x < n_grids & cell_pos.x >= 0 &
	//         cell_pos.y < n_grids & cell_pos.y >= 0 &
	//         cell_pos.z < n_grids & cell_pos.z >= 0) {
	//         res = cell_pos.x + n_grids * (cell_pos.y + n_grids * cell_pos.z);
	//     };
	//     return res;
	// };

	// lazy_compile(solver().device(), clear_cell,
	//              [&](Int count, BufferInt count_in_cell) {
	//                  set_block_size(n_blocks);
	//                  grid_stride_loop(count,
	//                                   [&](Int i) noexcept {
	//                                       count_in_cell.write(i, 0);
	//                                   });
	//              });

	// count = part
	lazy_compile(device, ms_count_sort_cell_sum,
				 [&](Int count, Int n_grids, Float cell_size,
					 BufferFloat3 part_pos) {
		set_block_size(n_blocks);
		grid_stride_loop(count,
						 [&](Int i) noexcept {
			Float3 pos = part_pos.read(i);
			UInt c = pos_to_cell_index(pos, n_grids, cell_size);
			m_hash->write(i, Int(c));

			Int index = mp_cell_state->particle_count->atomic(c).fetch_add(1);
			m_index->write(i, index);
		});
	});

	// count = part
	// lazy_compile(solver().device(), copy_from_tmp,
	//              [&](Int count, BufferInt part_id, BufferFloat3 part_pos, BufferFloat3 part_vel) {
	//                  set_block_size(n_blocks);
	//                  grid_stride_loop(count,
	//                                   [&](Int i) noexcept {
	//                                       part_id.write(i, tmp_id->read(i));
	//                                       part_pos.write(i, tmp_pos->read(i));
	//                                       part_vel.write(i, tmp_vel->read(i));
	//                                   });
	//              });

	// // count = part
	// lazy_compile(solver().device(), count_sort_result,
	//              [&](Int count, Int n_grids, Float cell_size, BufferInt part_id, BufferFloat3 part_pos, BufferFloat3 part_vel) {
	//                  set_block_size(n_blocks);
	//                  grid_stride_loop(count,
	//                                   [&](Int i) noexcept {
	//                                       Float3 pos = part_pos.read(i);
	//                                       UInt c = pos_to_cell_index(pos, n_grids, cell_size);

	//                                       Int index = m_index->read(i) + m_cell->particle_offset->read(c);
	//                                       m_index->write(i, index);

	//                                       // Sort
	//                                       tmp_pos->write(index, part_pos.read(i));
	//                                       tmp_vel->write(index, part_vel.read(i));
	//                                       tmp_id->write(index, part_id.read(i));
	//                                   });
	//              });

	// //// count = cell
	// lazy_compile(solver().device(), cal_block,
	//              [&](Int count) {
	//                  set_block_size(n_blocks);
	//                  grid_stride_loop(count,
	//                                   [&](Int i) noexcept {
	//                                       Int part_count = m_cell->particle_count->read(i);

	//                                       // m_cell->task_count->write(c, ceil_int(part_count, int(N_THREADS))); // (pc + 31) / 32
	//                                       m_cell->task_count->write(i, floor_int(part_count, n_threads));//  pc /32
	//                                   });
	//              });

	// // count = part
	// lazy_compile(solver().device(), count_sort_cell_sum_2,
	//              [&](Int count, Int n_grids, Int num_cells, Float cell_size, BufferFloat3 part_pos) {
	//                  set_block_size(n_blocks);
	//                  grid_stride_loop(count,
	//                                   [&](Int p) noexcept {
	//                                       Float3 pos = part_pos.read(p);
	//                                       UInt c = pos_to_cell_index(pos, n_grids, cell_size);
	//                                       UInt hash = c;
	//                                       UInt task_count = m_cell->task_count->read(c);
	//                                       UInt part_offset = m_cell->particle_offset->read(c);
	//                                       $if(task_count * n_threads >= p - part_offset + 1)// Densy Particle
	//                                       {
	//                                           hash = hash + num_cells;
	//                                       };

	//                                       Int thread_index = m_cell->particle_count_hash->atomic(hash).fetch_add(1);
	//                                       m_thread_index->write(p, thread_index);
	//                                       m_hash->write(p, Int(hash));
	//                                   });
	//              });

	// // count = part
	// lazy_compile(solver().device(), count_sort_result_2,
	//              [&](Int count) {
	//                  set_block_size(n_blocks);
	//                  grid_stride_loop(count,
	//                                   [&](Int p) noexcept {
	//                                       Int hash = m_hash->read(p);
	//                                       Int c = hash;

	//                                       Int thread_index = m_thread_index->read(p) + m_cell->particle_offset_hash->read(c);
	//                                       m_thread_index->write(p, thread_index);
	//                                       m_thread_to_data_index->write(thread_index, p);
	//                                   });
	//              });

	// // count = part
	// lazy_compile(solver().device(), find_middle_value,
	//              [&](Int count, Int n_particles, Int num_cells, Int n_hashs) {
	//                  set_block_size(n_blocks);
	//                  grid_stride_loop(
	//                      count,
	//                      [&](Int t) noexcept {
	//                          UInt thread_x = thread_id().x;
	//                          luisa::compute::Shared<int> shared_hash{n_blocks};

	//                          Int p = m_thread_to_data_index->read(t);
	//                          Int self_hash = m_hash->read(p);

	//                          // Shared MEM
	//                          $if(t == 0) {
	//                              m_data_gpu->write(0, n_particles);
	//                              m_data_gpu->write(1, ceil_int(n_particles, n_blocks_int));
	//                          };
	//                          $if(thread_x < UInt(n_blocks_int - 1)) { shared_hash[thread_x + 1] = self_hash; };
	//                          $if(t > 0 & thread_x == 0u) {
	//                              Int prior_p = m_thread_to_data_index->read(t - 1);
	//                              shared_hash[0] = m_hash->read(prior_p);
	//                          };

	//                          sync_block();// block sync

	//                          Int prior_hash;
	//                          $if(t == 0) { prior_hash = def(n_hashs); }
	//                          $else { prior_hash = shared_hash[thread_x]; };

	//                          $if(self_hash != prior_hash) {
	//                              $if(self_hash >= num_cells) {
	//                                  $if(t == 0 | prior_hash < num_cells) {
	//                                      m_data_gpu->write(0, t);
	//                                      m_data_gpu->write(1, ceil_int(t, n_blocks_int));
	//                                  };
	//                              };
	//                          };
	//                      },
	//                      [&](Int t) {},
	//                      [&](Int t) {
	//                          sync_block();
	//                      });
	//              });

	// // count = cell
	// lazy_compile(solver().device(), arrange_task,
	//              [&](Int count, Int num_cells) {
	//                  set_block_size(n_blocks);
	//                  grid_stride_loop(count,
	//                                   [&](Int c) noexcept {
	//                                       Int task_offset = m_cell->task_offset->read(c);
	//                                       Int task_count = m_cell->task_count->read(c);
	//                                       Int part_count = m_cell->particle_count->read(c);
	//                                       Int part_offset = 0;
	//                                       $for(i, task_offset, task_offset + task_count) {
	//                                           Int count = n_threads;
	//                                           $if(part_count - count < 0) { count = part_count; };
	//                                           part_count = part_count - count;
	//                                           m_task->cell_index->write(i, c);
	//                                           m_task->particle_count->write(i, count);
	//                                           m_task->particle_offset->write(i, part_offset);
	//                                           part_offset = part_offset + count;
	//                                       };

	//                                       // Current number of Tasks
	//                                       $if(c == num_cells - 1) {
	//                                           Int p_bt_offset = m_data_gpu->read(1);
	//                                           Int p_num_task = task_offset + task_count;
	//                                           Int p_blocks = (p_num_task * n_threads + n_blocks_int - 1) / n_blocks_int;
	//                                           Int p_num_thread = (p_blocks + p_bt_offset) * n_blocks_int;
	//                                           m_data_gpu->write(2, p_num_task);
	//                                           m_data_gpu->write(3, p_num_thread);
	//                                       };
	//                                   });
	//              });
}

}// namespace sail::inno::sph