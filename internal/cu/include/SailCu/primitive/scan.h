#pragma once
/**
 * @file primitive/scan.h
 * @brief The Parallel Scan
 * @date 2024-03-29
 * @author sailing-innocent
*/

#include "SailCu/config.h"

namespace sail::cu {

void SAIL_CU_API exclusive_scan(const int* d_temp_storage, int& temp_storage_size, const int* in_arr, int* out_arr, const int N);

}// namespace sail::cu