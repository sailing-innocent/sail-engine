#pragma once

#include <array>
#include <luisa/dsl/sugar.h>

namespace sail::inno {

inline luisa::float4x4 arr16_mat44(std::array<float, 16>& buf) {
	luisa::float4x4 mat;
	for (int i = 0; i < 16; i++) {
		mat[i / 4][i % 4] = buf[i];
	}
	return mat;
}

}// namespace sail::inno