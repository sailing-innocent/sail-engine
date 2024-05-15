#pragma once
/**
 * @file cub_wrapper.h
 * @brief the warpper of cub
 * @author sailing-innocent
 * @date 2024-05-05
 */

#include "SailCu/config.h"

namespace sail::cu {

/**
 * @brief the warpper of cub::DeviceScan::InclusiveSum
 * @param d_in the input array
 * @param d_out the output array
 * @param N the size of the array
 */

void SAIL_CU_API cub_inclusive_sum(int* d_in, int* d_out, int N);

}// namespace sail::cu