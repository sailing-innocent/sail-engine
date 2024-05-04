#pragma once
/**
 * @file SailCu/dummy.h
 * @brief Basic CUDA operations
 * @date 2023-10-04
 * @author sailing-innocent
*/
#include "SailBase/config.h"

namespace sail::cu {

void SAIL_CU_API cuda_inc(int* d_array, const int N);
void SAIL_CU_API cuda_add(int* d_array_a, int* d_array_b, int* d_array_c, const int N);

}// namespace sail::cu
