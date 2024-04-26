#include "SailLLVM/k_compiler/compiler.h"

int main() {
	sail::llvm::Compiler compiler;
	compiler.compile("def foo(x) x + 1");
	return 0;
}