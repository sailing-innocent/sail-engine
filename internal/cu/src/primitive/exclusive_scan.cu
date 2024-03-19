/**
 * @file primitive/scan.h
 * @brief The Parallel Scan
 * @date 2024-03-29
 * @author sailing-innocent
*/
#include "SailCu/primitive/scan.h"
#include <cmath>

namespace sail::cu {

#define BLOCK_SIZE 256

void exclusive_scan(const int* d_temp_storage, int& temp_storage_size, const int* in_arr, int* out_arr, const int N) {
	int num_blocks;
	auto max_N_elements = N;
	auto N_elements = max_N_elements;
	int level = 0;
	if (!d_temp_storage) {
		// allocate required temporary storage
		temp_storage_size = 0;
		do {
			num_blocks = (N_elements + BLOCK_SIZE - 1) / BLOCK_SIZE;
			num_blocks = 1 > num_blocks ? 1 : num_blocks;
			if (num_blocks > 1) {
				level++;
				temp_storage_size += num_blocks;
			}
			N_elements = num_blocks;// size of partial sum of this level
		} while (N_elements > 1);
		temp_storage_size += 1;
		return;// early return
	}

	// down-sweep
	// up-sweep
	// recursively
}

}// namespace sail::cu