#pragma once
/**
 * @file demo/simple2d.h
 * @brief Some Simple CUDA shader
 * @date 2024-03-20
 * @author sailing-innocent
*/
#include "SailCu/config.h"
namespace sail::cu {

class SAIL_CU_API Simple2DShader {
public:
	static void sine_wave(float* pixels, float t, int h, int w);
};

}// namespace sail::cu