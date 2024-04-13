/**
 * @file helper/device_parallel/device_parallel_scan_impl.cpp
 * @author sailing-innocent
 * @brief The device parallel api implementation
 * @date 2024-04-13
 */
#include "SailInno/helper/device_parallel.h"
#include "SailInno/util/math/calc.h"// for imax

using namespace luisa;
using namespace luisa::compute;

namespace sail::inno {

void DeviceParallel::prescan_array_recursive_int(
	luisa::compute::CommandList& cmdlist,
	BufferView<IntType> temp_storage,
	BufferView<IntType> arr_in, BufferView<IntType> arr_out,
	size_t num_elements, int offset, int level) noexcept {
	using namespace inno::math;
	int block_size = m_block_size;
	int num_blocks = imax(1, (int)ceil((float)num_elements / (2.0f * block_size)));
	int num_threads;

	if (num_blocks > 1) {
		num_threads = block_size;
	} else if (is_power_of_two(num_elements)) {
		num_threads = num_elements / 2;
	} else {
		num_threads = floor_pow_2(num_elements);
	}

	int num_elements_per_block = num_threads * 2;
	int num_elements_last_block =
		num_elements - (num_blocks - 1) * num_elements_per_block;
	int num_threads_last_block = imax(1, num_elements_last_block / 2);
	int np2_last_block = 0;
	int shared_mem_last_block = 0;

	if (num_elements_last_block != num_elements_per_block) {
		// NOT POWER OF 2
		np2_last_block = 1;
		if (!is_power_of_two(num_elements_last_block)) {
			num_threads_last_block = floor_pow_2(num_elements_last_block);
		}
	}

	size_t size_elements = temp_storage.size() - offset;
	BufferView<IntType> temp_buffer_level =
		temp_storage.subview(offset, size_elements);

	// execute the scan

	if (num_blocks > 1) {
		// recursive
		cmdlist << (*ms_prescan_int)(true, false,
									 arr_in, arr_out, temp_buffer_level,
									 num_elements_per_block, 0, 0)
					   .dispatch(block_size * (num_blocks - np2_last_block));

		if (np2_last_block) {
			// Last Block
			cmdlist << (*ms_prescan_int)(
						   true, true,
						   arr_in, arr_out, temp_buffer_level,
						   num_elements_last_block, num_blocks - 1,
						   num_elements - num_elements_last_block)
						   .dispatch(block_size);
		}

		prescan_array_recursive_int(
			cmdlist,
			temp_buffer_level, temp_buffer_level,
			temp_buffer_level,
			num_blocks, num_blocks, level + 1);

		cmdlist << (*ms_uniform_add_int)(
					   arr_out, temp_buffer_level,
					   num_elements - num_elements_last_block,
					   0, 0)
					   .dispatch(block_size * (num_blocks - np2_last_block));

		if (np2_last_block) {
			cmdlist << (*ms_uniform_add_int)(
						   arr_out, temp_buffer_level,
						   num_elements_last_block, num_blocks - 1,
						   num_elements - num_elements_last_block)
						   .dispatch(block_size);
		}
	} else if (is_power_of_two(num_elements)) {
		// non-recursive
		cmdlist << (*ms_prescan_int)(
					   false, false,
					   arr_in, arr_out, temp_buffer_level,
					   num_elements, 0, 0)
					   .dispatch(block_size);
	} else {
		// non-recursive
		cmdlist << (*ms_prescan_int)(
					   false, true,
					   arr_in, arr_out, temp_buffer_level,
					   num_elements, 0, 0)
					   .dispatch(block_size);
	}
}

}// namespace sail::inno