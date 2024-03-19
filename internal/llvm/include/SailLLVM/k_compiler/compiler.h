#pragma once
#include "SailLLVM/config.h"
#include <memory>

namespace sail::llvm {

struct SAIL_LLVM_API Compiler {
	int dummy = 0;
	int return_one() const;
};

}// namespace sail::llvm