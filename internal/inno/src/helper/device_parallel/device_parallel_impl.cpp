/**
 * @file helper/device_parallel_api.cpp
 * @author sailing-innocent
 * @brief The device parallel api implementation
 * @date 2023-12-28
 */
#include "SailInno/helper/device_parallel.h"
#include "SailInno/util/math/calc.h"// for imax

using namespace luisa;
using namespace luisa::compute;

// API

namespace sail::inno {

void DeviceParallel::create(Device& device) {
	// argument will not change after create
	int num_elements_per_block = m_block_size * 2;
	int extra_space = num_elements_per_block / m_num_banks;
	m_shared_mem_size = (num_elements_per_block + extra_space);
	LUISA_INFO("Smem Size: {}", m_shared_mem_size);
	compile<int>(device);
	compile<float>(device);
}

void DeviceParallel::scan_inclusive_sum(
	size_t& temp_storage_size,
	BufferView<IntType> d_in,
	BufferView<IntType> d_out,
	size_t num_item) {
	get_temp_size(temp_storage_size, num_item);
}

void DeviceParallel::scan_inclusive_sum(
	CommandList& cmdlist,
	BufferView<IntType> temp_buffer,
	BufferView<IntType> d_in,
	BufferView<IntType> d_out, size_t num_item) {
	size_t temp_storage_size = 0;
	get_temp_size(temp_storage_size, num_item);
	LUISA_ASSERT(temp_storage_size <= temp_buffer.size(), "temp_buffer size is not enough");
	// prescan_array_recursive_int(
	// 	cmdlist,
	// 	temp_buffer, d_in, d_out,
	// 	num_item, 0, 0);
}

}// namespace sail::inno

// Core Util

namespace sail::inno {

void DeviceParallel::get_temp_size(size_t& temp_storage_size, size_t num_items) {
	auto block_size = m_block_size;
	unsigned int max_num_elements = num_items;
	temp_storage_size = 0;
	unsigned int num_elements = max_num_elements;// input segment size
	int level = 0;

	do {
		// output segment size
		unsigned int num_blocks = math::imax(1, (int)ceil((float)num_elements / (2.f * block_size)));
		if (num_blocks > 1) {
			level++;
			temp_storage_size += num_blocks;
		}
		num_elements = num_blocks;
	} while (num_elements > 1);
	temp_storage_size += 1;
}

}// namespace sail::inno