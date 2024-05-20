#pragma once
/**
 * @file types.h
 * @brief The Particle IR Types
 * @author sailing-innocent
 * @date 2024-05-19
 */
#include <luisa/luisa-compute.h>

namespace sail::inno::pir {

struct BufferRegisterDescriptor {
	luisa::string name;
	luisa::compute::Type* type;
};

struct VariableRegisterDescriptor {
	luisa::string name;
	luisa::compute::Type* type;
};

class ParticleIR {
};

}// namespace sail::inno::pir