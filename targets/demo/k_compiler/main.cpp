#include "SailLLVM/k_compiler/compiler.h"

int main() {
	sail::llvm::Compiler compiler;
	compiler.compile("def foo(x)\n x + 1\0");
	return 0;
}