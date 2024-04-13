/**
 * @file helper/device_parallel/device_parallel_reduce_impl.cpp
 * @author sailing-innocent
 * @brief The device parallel api implementation
 * @date 2023-12-28
 */
#include "SailInno/helper/device_parallel.h"
#include "SailInno/util/math/calc.h"// for imax

using namespace luisa;
using namespace luisa::compute;

namespace sail::inno {

void DeviceParallel::reduce_array_recursive_int(
	luisa::compute::CommandList& cmdlist,
	BufferView<IntType> temp_storage,
	BufferView<IntType> arr_in,
	BufferView<IntType> arr_out,
	int num_elements,
	int offset, int level, int op) noexcept {
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

	// execute the reduce
	if (num_blocks > 1) {
		if (np2_last_block) {
		}

		// After scanning all the sub-blocks, we are mostly done.  But now we
		// need to take all of the last values of the sub-blocks and scan those.
		// This will give us a new value that must be added to each block to
		// get the final results.
		// recursive (CPU) call
		// Next level's offset = Cur level's numBlocks
	} else {
		// cmdlist << (*reduceInt)(arr_in, temp_buffer_level, numElements, 0, 0, op).dispatch(blockSize);// Due to block_size, numThreads -> blockSize
		// Store Answer arr_out[0] = temp_buffer_level[0]
		LUISA_ASSERT(arr_out.size() >= 1, "Resize the arr_out Buffer.");
		cmdlist << arr_out.copy_from(temp_buffer_level);
	}
}

}// namespace sail::inno