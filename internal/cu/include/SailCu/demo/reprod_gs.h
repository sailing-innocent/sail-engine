#pragma once
/**
 * @file demo/reprod_gs.h
 * @brief The CUDA based Gaussian Splatting Reproduction
 * @date 2024-04-24
 * @author sailing-innocent
*/

#include "SailCu/config.h"
#include <span>

namespace sail::cu {

class SAIL_CU_API ReprodGs {
public:
	ReprodGs() = default;
	~ReprodGs() = default;
	void debug_img(int w, int h, std::span<float> h_out) noexcept;
};

}// namespace sail::cu