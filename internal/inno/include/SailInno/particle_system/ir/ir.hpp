#pragma once
/**
 * @file SailInno/particle_system/ir.hpp
 * @author sailing-innocent
 * @date 2024-03-30
 * @brief The Particle IR
*/

namespace sail::inno {

enum class ModuleKind {
	Block,
	Function,
	Kernel
};

// TODO
enum class Primitive {
	Bool,
	Int
};
struct NodeRef {
	size_t _id;
	bool operator==(const NodeRef& other) const {
		return _id == other._id;
	}
};

}// namespace sail::inno