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

}// namespace sail::cu
