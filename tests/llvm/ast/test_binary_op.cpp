#include "test_util.h"

#include "SailLLVM/k_compiler/ast/expr.h"
#include <memory>

namespace sail::test {
using namespace sail::llvm;

int test_ast_binary_op() {
	auto LHS = std::make_unique<NumberExprAST>(2.0);
	auto RHS = std::make_unique<NumberExprAST>(3.0);
	auto BinOp = std::make_unique<BinaryOpAST>('+', std::move(LHS), std::move(RHS));
	return 0;
}

}// namespace sail::test

TEST_SUITE("llvm::ast") {
	TEST_CASE("binary_op") {
		CHECK(sail::test::test_ast_binary_op() == 0);
	}
}