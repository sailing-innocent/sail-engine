#pragma once
/**
 * @author Oncle-Ha
 * @date 2023-04-06
 */

#include <luisa/core/basic_types.h>
#include <luisa/runtime/buffer.h>
#include <luisa/runtime/device.h>
#include <luisa/core/logging.h>

#include "core/lazy/lazy.h"

#include "sph_executor.h"
#include "fluid_particles.h"

#include "packages/buffer/buffer_filler.h"
#include "packages/parallel/device_parallel.h"

namespace inno::csigsph {
class SPHSolver;

static constexpr auto INVALID_CELL_INDEX = -1;
static constexpr float PI = 3.1415926f;

class Neighbor : public SPHExecutor {
    template<typename T>
    using U = luisa::unique_ptr<T>;

    template<typename T>
    using Buffer = luisa::compute::Buffer<T>;

    friend class SPHSolver;

public:
    class Task {
        friend class Neighbor;

    public:
        Buffer<int> particle_offset;// offset in cell
        Buffer<int> particle_count;
        Buffer<int> cell_index;

    private:
        void allocate(luisa::compute::Device &device, size_t size) {
            particle_offset = device.create_buffer<int>(size);
            particle_count = device.create_buffer<int>(size);
            cell_index = device.create_buffer<int>(size);
        }
    };

    class Cell {
        friend class Neighbor;

    public:
        Buffer<int> particle_offset;// data
        Buffer<int> particle_count;
        Buffer<int> particle_offset_hash;// hash
        Buffer<int> particle_count_hash;
        Buffer<int> task_count;
        Buffer<int> task_offset;

    private:
        void allocate(luisa::compute::Device &device, size_t size) {
            particle_offset = device.create_buffer<int>(size);
            particle_count = device.create_buffer<int>(size);
            particle_offset_hash = device.create_buffer<int>(size << 1);
            particle_count_hash = device.create_buffer<int>(size << 1);
            task_count = device.create_buffer<int>(size);
            task_offset = device.create_buffer<int>(size);
        }
    };

    // for lcub
    class ScanTempStorage {
        friend class Neighbor;

    public:
        Buffer<int> temp_storage;

    private:
        void allocate(luisa::compute::Device &device, size_t size) {
            temp_storage = device.create_buffer<int>(size);
        }
    };

    Neighbor(SPHSolver &solver) noexcept;

    Buffer<int> m_hash;                // cell_index (+ N_CELL)
    Buffer<int> m_cell_inex;           // cell_index
    Buffer<int> m_index;               // index in data (subscript)
    Buffer<int> m_thread_index;        // index in thread
    Buffer<int> m_thread_to_data_index;// index_thread -> index_data

    Buffer<int> tmp_id;
    Buffer<float3> tmp_pos;
    Buffer<float3> tmp_vel;

    Buffer<int> m_data_gpu;//[m_middle, m_bt_offset, m_num_task, m_num_thread]
    luisa::vector<int> m_data_cpu;

    int m_middle = 0;
    int m_bt_offset = 0;
    int m_num_task = 0;
    int m_num_thread_up = 0;

    float m_cell_size = 0.f;// cell size >= h_fac
    int m_num_grids = 0;
    int m_num_cells = 0;
    int m_num_tasks = 0;
    int m_num_hashs = 0;

    Task &task() noexcept { return *m_task; }
    Cell &cell() noexcept { return *m_cell; }
    ScanTempStorage &scan_temp() noexcept { return *m_scan_temp; }

    auto size() const noexcept { return m_size; }
    void solve(luisa::compute::CommandList &cmdlist) noexcept;

private:
    void create() noexcept;

    void compile() noexcept;
    void reset() noexcept;

    int get_thread_up(int x) noexcept;

    // void init_upload(CommandList &cmdlist) noexcept;

    void allocate(luisa::compute::Device &device, size_t size) noexcept;

    void after_solve() noexcept;

    size_t m_size = 0;
    size_t m_capacity = 0;

    // U<sphere::primitive::DeviceParallel> m_scan;
    U<Task> m_task;
    U<Cell> m_cell;
    U<ScanTempStorage> m_scan_temp;

    Global1D<int, Buffer<int>>::UShader clear_cell;
    Global1D<int, int, float, Buffer<float3>>::UShader count_sort_cell_sum;
    Global1D<int, Buffer<int>, Buffer<float3>, Buffer<float3>>::UShader copy_from_tmp;
    Global1D<int, int, float, Buffer<int>, Buffer<float3>, Buffer<float3>>::UShader count_sort_result;
    Global1D<int>::UShader cal_block;
    Global1D<int, int, int, float, Buffer<float3>>::UShader count_sort_cell_sum_2;
    Global1D<int>::UShader count_sort_result_2;
    Global1D<int, int, int, int>::UShader find_middle_value;
    Global1D<int, int>::UShader arrange_task;
};
}// namespace inno::csigsph