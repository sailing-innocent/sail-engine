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
	compile<int>(device);
	compile<float>(device);
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