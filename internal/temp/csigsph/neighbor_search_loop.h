#pragma once
/**
 * @author Oncle-Ha
 * @date 2023-04-06
 */
#include <luisa/dsl/printer.h>
#include <luisa/dsl/syntax.h>
#include <luisa/dsl/sugar.h>

#include "core/package/package.h"
#include "core/dsl/scope.h"

#include "neighbor.h"

namespace inno::csigsph {

using SMEM_int = luisa::compute::Shared<int>;
using SMEM_int_ptr = luisa::shared_ptr<luisa::compute::Shared<int>>;
using SMEM_float = luisa::compute::Shared<float>;
using SMEM_float_ptr = luisa::shared_ptr<luisa::compute::Shared<float>>;
using SMEM_float3 = luisa::compute::Shared<luisa::float3>;
using SMEM_float3_ptr = luisa::shared_ptr<luisa::compute::Shared<luisa::float3>>;

class Neighbor;
class FluidParticles;

inline luisa::compute::Int3 cell_index_to_ijk_neig(
    luisa::compute::UInt index, 
    const luisa::compute::Int &n_grids) noexcept {
    luisa::compute::Int z = luisa::compute::Int(index / (n_grids * n_grids));
    index = index % (n_grids * n_grids);
    luisa::compute::Int y = luisa::compute::Int(index / (n_grids));
    index = index % (n_grids);
    luisa::compute::Int x = luisa::compute::Int(index);
    return luisa::compute::make_int3(x, y, z);
};

inline luisa::compute::Int cell_pos_to_cell_index_neig(
    luisa::compute::Int3 &cell_pos, 
    const luisa::compute::Int &n_grids) noexcept {
    luisa::compute::Int res = luisa::compute::def(INVALID_CELL_INDEX);
    $if(cell_pos.x < n_grids & cell_pos.x >= 0 &
        cell_pos.y < n_grids & cell_pos.y >= 0 &
        cell_pos.z < n_grids & cell_pos.z >= 0) {
        res = cell_pos.x + n_grids * (cell_pos.y + n_grids * cell_pos.z);
    };
    return res;
};

inline auto ijk_to_cell_index_neig = [](
    luisa::compute::UInt3 &coord, 
    const luisa::compute::Int &n_grids) noexcept {
    using T = luisa::compute::vector_expr_element_t<decltype(coord)>;
    auto p = luisa::compute::clamp(coord, 0u, luisa::compute::UInt(n_grids - 1));
    return p.x + p.y * n_grids + p.z * n_grids * n_grids;
};

inline auto pos_to_cell_index_neig = [](luisa::compute::Float3 &pos, const luisa::compute::Int &n_grids, const luisa::compute::Float &cell_size) noexcept {
    luisa::compute::UInt3 coord;
    for (auto i = 0; i < 3; i++) coord[i] = luisa::compute::UInt(pos[i] / cell_size);
    return ijk_to_cell_index_neig(coord, n_grids);
};

inline void initialize(
    Neighbor &neighbor,
    SMEM_int_ptr cell_offset,
    SMEM_int_ptr cell_count,
    luisa::compute::Int3 cell_pos,
    luisa::compute::Int *current_cell_index,
    luisa::compute::Int *offset_in_cell,
    const luisa::compute::Int &n_grids,
    const luisa::compute::Int &n_threads) {
    using namespace luisa;
    using namespace luisa::compute;

    UInt idx = thread_id().x;
    Int bi = Int(idx / n_threads);// bi <= 1
    Int bj = Int(idx % n_threads);
    auto &p_cell = neighbor.cell();

    $if(bj < 9) {

        Int kk = bi * 9 + bj;

        Int3 neighbor_pos = cell_pos + make_int3(-1, bj % 3 - 1, bj / 3 % 3 - 1);
        $if(neighbor_pos.y >= 0 & neighbor_pos.y < n_grids &
            neighbor_pos.z >= 0 & neighbor_pos.z < n_grids) {
            // index continuous along X-axis
            Int nid_left, nid_mid, nid_right;
            Int part_offset, part_count = 0;
            nid_left = cell_pos_to_cell_index_neig(neighbor_pos, n_grids);
            neighbor_pos.x = neighbor_pos.x + 1;
            nid_mid = cell_pos_to_cell_index_neig(neighbor_pos, n_grids);
            neighbor_pos.x = neighbor_pos.x + 1;
            nid_right = cell_pos_to_cell_index_neig(neighbor_pos, n_grids);
            $if(nid_left != INVALID_CELL_INDEX) {
                part_offset = p_cell.particle_offset->read(nid_left);
                part_count = p_cell.particle_count->read(nid_left);
            }
            $else {
                part_offset = p_cell.particle_offset->read(nid_mid);
            };
            part_count = part_count + p_cell.particle_count->read(nid_mid);
            $if(nid_right != INVALID_CELL_INDEX) { part_count += p_cell.particle_count->read(nid_right); };

            (*cell_offset)[kk] = part_offset;
            (*cell_count)[kk] = part_count;
        }
        $else {
            (*cell_offset)[kk] = 0;
            (*cell_count)[kk] = 0;
        };
    };
    *current_cell_index = 9 + bi * 9;
    *offset_in_cell = 0;

    $for(i, bi * 9, bi * 9 + 9) {
        $if(0 != (*cell_count)[i]) {
            *current_cell_index = i;
            $break;
        };
    };
}

template<typename LoadSmemF>
inline luisa::compute::Int read_32_data(
    SMEM_float3_ptr &part_pos,
    SMEM_float3_ptr &part_vel,
    SMEM_float_ptr &part_w,
    SMEM_int_ptr &cell_offset,
    SMEM_int_ptr &cell_count,
    luisa::compute::Int *current_cell_index,
    luisa::compute::Int *offset_in_cell,
    LoadSmemF &&f_load,
    const luisa::compute::Int &n_threads) {
    using namespace luisa;
    using namespace luisa::compute;
    UInt idx = thread_id().x;
    ;
    Int bi = Int(idx / n_threads);// bi <= 1
    Int bj = Int(idx % n_threads);

    Int curr_cell_index = *current_cell_index;

    Int num_read = 0;

    $if(9 + bi * 9 > curr_cell_index) {
        Int offset = *offset_in_cell;
        Int remain_nump = (*cell_count)[curr_cell_index] - offset;

        $if(remain_nump > n_threads) { num_read = n_threads; }
        $else { num_read = remain_nump; };

        $if(num_read > bj) {
            UInt read_idx = UInt((*cell_offset)[curr_cell_index] + offset + bj);
            f_load(idx, read_idx, part_pos, part_vel, part_w);
        };

        $if(remain_nump > n_threads) { *offset_in_cell = offset + n_threads; }
        $else {
            Int next_cell_idx = curr_cell_index + 1;
            $while(next_cell_idx < 9 + bi * 9) {
                $if(0 != (*cell_count)[next_cell_idx]) { $break; };
                next_cell_idx = next_cell_idx + 1;
            };
            *current_cell_index = next_cell_idx;
            *offset_in_cell = 0;
        };
    };
    // neighbors read complete
    return num_read;
}

template<typename BeginF, typename LoopF, typename EndF>
inline void neig_search(
    Neighbor &neighbor,
    FluidParticles &particles,
    luisa::compute::Int &self_index,
    BeginF &&f_read,
    LoopF &&f_loop,
    EndF &&f_write,
    const luisa::compute::Int &n_grids,
    const luisa::compute::Int &n_threads,
    const luisa::compute::Int &n_cta,
    const luisa::compute::Float &cell_size) {
    using namespace luisa;
    using namespace luisa::compute;

    auto &p_cell = neighbor.cell();

    UInt self_p = UInt(self_index);
    Float3 pos_a;
    Float3 vel_a;
    Float w_a;
    Float3 res = make_float3(0.f);
    f_read(self_p, pos_a, vel_a, w_a);

    // The grid construction depends on the old position.
    Float3 pos_old = particles.m_pos->read(self_p);
    auto c = pos_to_cell_index_neig(pos_old, n_grids, cell_size);
    auto cell_pos = cell_index_to_ijk_neig(UInt(c), n_grids);

    for (int i = 0; i < 9; ++i) {
        Int3 neighbor_pos = cell_pos + luisa::make_int3(-1, i % 3 - 1, i / 3 % 3 - 1);
        $if(neighbor_pos.y >= 0 & neighbor_pos.y < n_grids &
            neighbor_pos.z >= 0 & neighbor_pos.z < n_grids) {
            // index continuous along X-axis
            Int nid_left, nid_mid, nid_right;
            Int part_offset, part_count = 0;
            nid_left = cell_pos_to_cell_index_neig(neighbor_pos, n_grids);
            neighbor_pos.x = neighbor_pos.x + 1;
            nid_mid = cell_pos_to_cell_index_neig(neighbor_pos, n_grids);
            neighbor_pos.x = neighbor_pos.x + 1;
            nid_right = cell_pos_to_cell_index_neig(neighbor_pos, n_grids);
            $if(nid_left != INVALID_CELL_INDEX) {
                part_offset = p_cell.particle_offset->read(nid_left);
                part_count = p_cell.particle_count->read(nid_left);
            }
            $else {
                part_offset = p_cell.particle_offset->read(nid_mid);
            };
            part_count = part_count + p_cell.particle_count->read(nid_mid);
            $if(nid_right != INVALID_CELL_INDEX) { part_count += p_cell.particle_count->read(nid_right); };

            //search [part_offset, part_offset + part_count)
            $for(neighbor_p, part_offset, part_offset + part_count) {
                // compute content
                UInt tmp = UInt(neighbor_p);
                Float3 pos_b, vel_b;
                Float w_b;
                f_read(tmp, pos_b, vel_b, w_b);
                f_loop(pos_a, pos_b, vel_a, vel_b, w_a, w_b, res);
            };
        };
    }
    // write data
    f_write(self_p, res);
}

template<typename BeginF, typename NewSmemF, typename LoadSmemF, typename LoopF, typename EndF>
inline void task_search(
    Neighbor &neighbor,
    FluidParticles &particles,
    const luisa::compute::Int n_grids,
    const luisa::compute::Int n_threads,
    const luisa::compute::Int n_cta,
    const luisa::compute::Float cell_size,
    BeginF &&f_read, 
    NewSmemF &&f_new, 
    LoadSmemF &&f_load, 
    LoopF &&f_loop, 
    EndF &&f_write) {
    using namespace luisa;
    using namespace luisa::compute;

    auto &p_cell = neighbor.cell();
    auto &p_task = neighbor.task();

    auto t = dispatch_id().x;// index of global thread
    auto block_x = block_id().x;
    auto thread_x = thread_id().x;// index of block thread
    Int middle = neighbor.m_data_gpu->read(0);
    Int offset_b = neighbor.m_data_gpu->read(1);
    Int num_thread_end = neighbor.m_data_gpu->read(3);
    $if(block_x < UInt(offset_b))// Sparse
    {
        $if(t < UInt(middle)) {
            auto self_index = neighbor.m_thread_to_data_index->read(t);
            neig_search(
                neighbor, particles,
                self_index,
                f_read, f_loop, f_write,
                n_grids, n_threads, n_cta, cell_size);
        };
    }
    $elif(t < UInt(num_thread_end))// Densy
    {
        Int densy_block = Int(block_x - offset_b);
        Int task_idx = densy_block * n_cta + thread_x / n_threads;
        Int num_task = Int(int(p_task.particle_offset.size()));

        $if(task_idx >= num_task) { task_idx = num_task - 1; };

        Int task_offset = p_task.particle_offset->read(task_idx);
        Int task_count = p_task.particle_count->read(task_idx);
        Int cell_id = p_task.cell_index->read(task_idx);
        Int self_index = p_cell.particle_offset->read(cell_id)// particle's index in data
                         + task_offset + Int(thread_x % n_threads);
        UInt self_p = UInt(self_index);
        Int temp_cell_end = p_cell.particle_offset->read(cell_id) +
                            p_cell.particle_count->read(cell_id);

        SMEM_int_ptr search_cell_offset_ptr = nullptr;
        SMEM_int_ptr search_cell_count_ptr = nullptr;

        Int offset_in_cell, current_cell_index;
        SMEM_float3_ptr search_part_pos_ptr = nullptr;
        SMEM_float3_ptr search_part_vel_ptr = nullptr;
        SMEM_float_ptr search_part_w_ptr = nullptr;

        // new SMEM
        f_new(search_part_pos_ptr, search_part_vel_ptr, search_part_w_ptr, search_cell_offset_ptr, search_cell_count_ptr);

        Float3 pos_a = make_float3(0.f);
        Float3 vel_a = make_float3(0.f);
        Float w_a = def(0.f);
        Int3 cell_ijk = cell_index_to_ijk_neig(cell_id, n_grids);

        // if self_index > temp_cell_end, it still need to read Data for other threads in the same block.
        // cell_ijk should be same in the same task.
        $if(self_index < temp_cell_end) {
            f_read(self_p, pos_a, vel_a, w_a);
        };

        initialize(
            neighbor,
            search_cell_offset_ptr, search_cell_count_ptr,
            cell_ijk,
            &current_cell_index, &offset_in_cell,
            n_grids, n_threads);
        sync_block();
        Float3 res = make_float3(0.f);
        $while(true) {
            Int r = 0;
            r = read_32_data(
                search_part_pos_ptr, search_part_vel_ptr, search_part_w_ptr,
                search_cell_offset_ptr, search_cell_count_ptr,
                &current_cell_index, &offset_in_cell,
                f_load,
                n_threads);

            sync_block();
            $if(r == 0) { $break; };
            Int bi_offset = Int((thread_x / n_threads) * n_threads);

            // Neighbor Search (include self_index)
            $for(i, bi_offset, bi_offset + r) {
                $if(self_index < temp_cell_end) {
                    Float3 pos_i;
                    Float3 vel_i;
                    Float w_i;
                    if (search_part_pos_ptr != NULL)
                        pos_i = (*search_part_pos_ptr)[i];
                    if (search_part_vel_ptr != NULL)
                        vel_i = (*search_part_vel_ptr)[i];
                    if (search_part_w_ptr != NULL)
                        w_i = (*search_part_w_ptr)[i];

                    // compute content
                    f_loop(pos_a, pos_i, vel_a, vel_i, w_a, w_i, res);
                };
            };
            sync_block();
        };

        $if(self_index < temp_cell_end) {
            // write data
            f_write(self_p, res);
        };
    };
}
}// namespace inno::csigsph
