/**
 * @author Oncle-Ha
 * @date 2023-04-06
 */

#include "fluid_particles.h"
#include "sph.h"
#include "neighbor.h"

// #include <deprecate/lcub/lcub.h>
// #include <backends/cuda/cuda_command_encoder.h>
// #include <lcub/device_scan.h>

// CORE IMPLEMENTATION
namespace inno::csigsph {
void Neighbor::compile() noexcept {
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
        for (auto i = 0; i < dim; i++) coord[i] = UInt(pos[i] / cell_size);
        return ijk_to_cell_index(coord, n_grids);
    };

    auto ceil_int = [&](auto a, auto b) noexcept { return (a + b - 1) / b; };
    auto floor_int = [&](auto a, auto b) noexcept { return a / b; };

    Callable cell_index_to_ijk = [&](UInt index, Int n_grids) noexcept {
        Int z = Int(index / (n_grids * n_grids));
        index = index % (n_grids * n_grids);
        Int y = Int(index / (n_grids));
        index = index % (n_grids);
        Int x = Int(index);
        return make_int3(x, y, z);
    };

    Callable cell_pos_to_cell_index = [&](UInt3 cell_pos, Int n_grids) noexcept {
        Int res = def(INVALID_CELL_INDEX);
        $if(cell_pos.x < n_grids & cell_pos.x >= 0 &
            cell_pos.y < n_grids & cell_pos.y >= 0 &
            cell_pos.z < n_grids & cell_pos.z >= 0) {
            res = cell_pos.x + n_grids * (cell_pos.y + n_grids * cell_pos.z);
        };
        return res;
    };

    lazy_compile(solver().device(), clear_cell,
                 [&](Int count, BufferInt count_in_cell) {
                     set_block_size(n_blocks);
                     grid_stride_loop(count,
                                      [&](Int i) noexcept {
                                          count_in_cell.write(i, 0);
                                      });
                 });

    // count = part
    lazy_compile(solver().device(), count_sort_cell_sum,
                 [&](Int count, Int n_grids, Float cell_size,
                     BufferFloat3 part_pos) {
                     set_block_size(n_blocks);
                     grid_stride_loop(count,
                                      [&](Int i) noexcept {
                                          Float3 pos = part_pos.read(i);
                                          UInt c = pos_to_cell_index(pos, n_grids, cell_size);
                                          m_hash->write(i, Int(c));

                                          Int index = m_cell->particle_count->atomic(c).fetch_add(1);
                                          m_index->write(i, index);
                                      });
                 });

    // count = part
    lazy_compile(solver().device(), copy_from_tmp,
                 [&](Int count, BufferInt part_id, BufferFloat3 part_pos, BufferFloat3 part_vel) {
                     set_block_size(n_blocks);
                     grid_stride_loop(count,
                                      [&](Int i) noexcept {
                                          part_id.write(i, tmp_id->read(i));
                                          part_pos.write(i, tmp_pos->read(i));
                                          part_vel.write(i, tmp_vel->read(i));
                                      });
                 });

    // count = part
    lazy_compile(solver().device(), count_sort_result,
                 [&](Int count, Int n_grids, Float cell_size, BufferInt part_id, BufferFloat3 part_pos, BufferFloat3 part_vel) {
                     set_block_size(n_blocks);
                     grid_stride_loop(count,
                                      [&](Int i) noexcept {
                                          Float3 pos = part_pos.read(i);
                                          UInt c = pos_to_cell_index(pos, n_grids, cell_size);

                                          Int index = m_index->read(i) + m_cell->particle_offset->read(c);
                                          m_index->write(i, index);

                                          // Sort
                                          tmp_pos->write(index, part_pos.read(i));
                                          tmp_vel->write(index, part_vel.read(i));
                                          tmp_id->write(index, part_id.read(i));
                                      });
                 });

    //// count = cell
    lazy_compile(solver().device(), cal_block,
                 [&](Int count) {
                     set_block_size(n_blocks);
                     grid_stride_loop(count,
                                      [&](Int i) noexcept {
                                          Int part_count = m_cell->particle_count->read(i);

                                          // m_cell->task_count->write(c, ceil_int(part_count, int(N_THREADS))); // (pc + 31) / 32
                                          m_cell->task_count->write(i, floor_int(part_count, n_threads));//  pc /32
                                      });
                 });

    // count = part
    lazy_compile(solver().device(), count_sort_cell_sum_2,
                 [&](Int count, Int n_grids, Int num_cells, Float cell_size, BufferFloat3 part_pos) {
                     set_block_size(n_blocks);
                     grid_stride_loop(count,
                                      [&](Int p) noexcept {
                                          Float3 pos = part_pos.read(p);
                                          UInt c = pos_to_cell_index(pos, n_grids, cell_size);
                                          UInt hash = c;
                                          UInt task_count = m_cell->task_count->read(c);
                                          UInt part_offset = m_cell->particle_offset->read(c);
                                          $if(task_count * n_threads >= p - part_offset + 1)// Densy Particle
                                          {
                                              hash = hash + num_cells;
                                          };

                                          Int thread_index = m_cell->particle_count_hash->atomic(hash).fetch_add(1);
                                          m_thread_index->write(p, thread_index);
                                          m_hash->write(p, Int(hash));
                                      });
                 });

    // count = part
    lazy_compile(solver().device(), count_sort_result_2,
                 [&](Int count) {
                     set_block_size(n_blocks);
                     grid_stride_loop(count,
                                      [&](Int p) noexcept {
                                          Int hash = m_hash->read(p);
                                          Int c = hash;

                                          Int thread_index = m_thread_index->read(p) + m_cell->particle_offset_hash->read(c);
                                          m_thread_index->write(p, thread_index);
                                          m_thread_to_data_index->write(thread_index, p);
                                      });
                 });

    // count = part
    lazy_compile(solver().device(), find_middle_value,
                 [&](Int count, Int n_particles, Int num_cells, Int n_hashs) {
                     set_block_size(n_blocks);
                     grid_stride_loop(
                         count,
                         [&](Int t) noexcept {
                             UInt thread_x = thread_id().x;
                             luisa::compute::Shared<int> shared_hash{n_blocks};

                             Int p = m_thread_to_data_index->read(t);
                             Int self_hash = m_hash->read(p);

                             // Shared MEM
                             $if(t == 0) {
                                 m_data_gpu->write(0, n_particles);
                                 m_data_gpu->write(1, ceil_int(n_particles, n_blocks_int));
                             };
                             $if(thread_x < UInt(n_blocks_int - 1)) { shared_hash[thread_x + 1] = self_hash; };
                             $if(t > 0 & thread_x == 0u) {
                                 Int prior_p = m_thread_to_data_index->read(t - 1);
                                 shared_hash[0] = m_hash->read(prior_p);
                             };

                             sync_block();// block sync

                             Int prior_hash;
                             $if(t == 0) { prior_hash = def(n_hashs); }
                             $else { prior_hash = shared_hash[thread_x]; };

                             $if(self_hash != prior_hash) {
                                 $if(self_hash >= num_cells) {
                                     $if(t == 0 | prior_hash < num_cells) {
                                         m_data_gpu->write(0, t);
                                         m_data_gpu->write(1, ceil_int(t, n_blocks_int));
                                     };
                                 };
                             };
                         },
                         [&](Int t) {},
                         [&](Int t) {
                             sync_block();
                         });
                 });

    // count = cell
    lazy_compile(solver().device(), arrange_task,
                 [&](Int count, Int num_cells) {
                     set_block_size(n_blocks);
                     grid_stride_loop(count,
                                      [&](Int c) noexcept {
                                          Int task_offset = m_cell->task_offset->read(c);
                                          Int task_count = m_cell->task_count->read(c);
                                          Int part_count = m_cell->particle_count->read(c);
                                          Int part_offset = 0;
                                          $for(i, task_offset, task_offset + task_count) {
                                              Int count = n_threads;
                                              $if(part_count - count < 0) { count = part_count; };
                                              part_count = part_count - count;
                                              m_task->cell_index->write(i, c);
                                              m_task->particle_count->write(i, count);
                                              m_task->particle_offset->write(i, part_offset);
                                              part_offset = part_offset + count;
                                          };

                                          // Current number of Tasks
                                          $if(c == num_cells - 1) {
                                              Int p_bt_offset = m_data_gpu->read(1);
                                              Int p_num_task = task_offset + task_count;
                                              Int p_blocks = (p_num_task * n_threads + n_blocks_int - 1) / n_blocks_int;
                                              Int p_num_thread = (p_blocks + p_bt_offset) * n_blocks_int;
                                              m_data_gpu->write(2, p_num_task);
                                              m_data_gpu->write(3, p_num_thread);
                                          };
                                      });
                 });
}
}// namespace inno::csigsph

// API IMPLEMENTATION
namespace inno::csigsph {
Neighbor::Neighbor(SPHSolver &solver) noexcept : SPHExecutor{solver} {
    m_task = luisa::make_unique<Task>();
    m_cell = luisa::make_unique<Cell>();
    m_scan_temp = luisa::make_unique<ScanTempStorage>();
}

void Neighbor::reset() noexcept {
    m_size = solver().particles().size();
    m_cell_size = solver().param().h_fac;
    // cell_size should not less than min_cell_size to avoid too many grids
    if (m_cell_size < solver().config().min_cell_size)
        m_cell_size = solver().config().min_cell_size;
    m_num_grids = (solver().config().world_size / m_cell_size) + 1;
    m_num_cells = m_num_grids * m_num_grids * m_num_grids;
    m_num_tasks = m_capacity / 32 + m_num_cells;
    m_num_hashs = m_num_cells * 2;
    m_num_thread_up = get_thread_up(m_size);// upper limit

    LUISA_INFO("Cell size:{}", m_cell_size);
    LUISA_INFO("Thread num:{}", m_num_thread_up);
}

void Neighbor::create() noexcept {
    // using namespace lcub;
    using namespace inno::primitive;
    reset();

    // allocate memory
    m_capacity = solver().config().n_capacity;
    int max_num_grids = (solver().config().world_size / solver().config().min_cell_size) + 1;
    int max_num_cells = max_num_grids * max_num_grids * max_num_grids;
    int max_num_tasks = m_capacity / 32 + max_num_cells;

    size_t num_cells = max_num_cells;
    size_t num_tasks = max_num_tasks;

    allocate(solver().device(), m_capacity);
    m_task->allocate(solver().device(), num_tasks);
    m_cell->allocate(solver().device(), num_cells);

    // get temp storage size
    size_t temp_storage_size = -1;
    solver().device_parallel().scan_exclusive_sum(temp_storage_size, m_cell->particle_count_hash, m_cell->particle_offset_hash, num_cells << 1);
    // create temp storage
    m_scan_temp->allocate(solver().device(), temp_storage_size);
}

void Neighbor::allocate(luisa::compute::Device &device, size_t size) noexcept {
    m_hash = device.create_buffer<int>(size);
    m_cell_inex = device.create_buffer<int>(size);
    m_index = device.create_buffer<int>(size);
    m_thread_index = device.create_buffer<int>(size);
    m_thread_to_data_index = device.create_buffer<int>(size);

    tmp_id = device.create_buffer<int>(size);
    tmp_pos = device.create_buffer<float3>(size);
    tmp_vel = device.create_buffer<float3>(size);

    // len = [4]
    m_data_gpu = device.create_buffer<int>(4);
    m_data_cpu.resize(4);
}
void Neighbor::solve(luisa::compute::CommandList &cmdlist) noexcept {
    // using namespace lcub;
    using namespace inno::primitive;

    int num_cells = m_num_cells;
    int num_particles = m_size;
    int n_grids = m_num_grids;
    int n_hashs = m_num_hashs;
    float cell_size = m_cell_size;
    auto &particles = solver().particles();

    // Count particles in each cell
    cmdlist << solver().filler().fill(m_cell->particle_count, 0)
            << solver().filler().fill(m_cell->task_count, 0)
            << (*count_sort_cell_sum)(num_particles, n_grids, cell_size, particles.m_pos).dispatch(num_particles);
    solver().device_parallel().scan_exclusive_sum(cmdlist, m_scan_temp->temp_storage, m_cell->particle_count, m_cell->particle_offset, num_cells);

    cmdlist << (*count_sort_result)(num_particles, n_grids, cell_size, particles.m_id, particles.m_pos, particles.m_vel).dispatch(num_particles)

            // Count tasks in each cell
            << (*cal_block)(num_cells).dispatch(num_cells)
            << (*copy_from_tmp)(num_particles, particles.m_id, particles.m_pos, particles.m_vel).dispatch(num_particles);
    solver().device_parallel().scan_exclusive_sum(cmdlist, m_scan_temp->temp_storage, m_cell->task_count, m_cell->task_offset, num_cells);

    // Count hash for each particle
    cmdlist << solver().filler().fill(m_cell->particle_count_hash, 0)
            << (*count_sort_cell_sum_2)(num_particles, n_grids, num_cells, cell_size, particles.m_pos).dispatch(num_particles);
    solver().device_parallel().scan_exclusive_sum(cmdlist, m_scan_temp->temp_storage, m_cell->particle_count_hash, m_cell->particle_offset_hash, num_cells << 1);
    
    cmdlist << (*count_sort_result_2)(num_particles).dispatch(num_particles)

            // Find dividing line between sparse and dense particles
            // [Sparse, Dense]
            << (*find_middle_value)(num_particles, num_particles, num_cells, n_hashs).dispatch(num_particles)
            // Arrange particles for each task
            << (*arrange_task)(num_cells, num_cells).dispatch(num_cells)
            << m_data_gpu.copy_to(m_data_cpu.data());
}

int Neighbor::get_thread_up(int x) noexcept {
    int block_int = int(solver().config().n_blocks);
    return (x / block_int + 2) * block_int;
}

// TODO remove
void Neighbor::after_solve() noexcept {
    m_middle = m_data_cpu[0];
    m_bt_offset = m_data_cpu[1];
    m_num_task = m_data_cpu[2];
    int n_blocks = solver().config().n_blocks;
    int p_blocks = (m_num_task * solver().config().n_threads + n_blocks - 1) / n_blocks;
    m_num_thread_up = (p_blocks + m_bt_offset) * n_blocks;
    LUISA_INFO("[Neighbor] midlle:{} offset:{} task:{} thread:{}", m_middle, m_bt_offset, m_num_task, m_num_thread_up);
}

}// namespace inno::csigsph