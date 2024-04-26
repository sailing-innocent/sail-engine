#pragma once
#include "SailLLVM/config.h"
#include <string>
#include <vector>
#include <map>
#include "ast/expr.h"

namespace sail::llvm {

// A State Machine Compiler
struct SAIL_LLVM_API Compiler {
	Compiler();
	enum struct Token : int {
		TOK_EOF = -1,
		TOK_DEF = -2,
		TOK_EXTERN = -3,
		TOK_IDENTIFIER = -4,
		TOK_NUMBER = -5
	};
	std::string m_identifier_str;// filled if TOK_IDENTIFIER
	double m_num_val;			 // filled if TOK_NUMBER
	int m_code_idx = 0;
	int m_code_size = 0;
	std::string m_code;
	void compile(std::string_view code) noexcept;
	unsigned char m_last_char = ' ';// for get_tok
	int m_cur_tok;
	[[nodiscard]] int get_tok(std::string_view code) noexcept;
	int get_next_tok() noexcept;
	std::map<char, int> m_bin_op_precedence;
	int get_tok_prec() noexcept;
	// parser
	void handle_difinition() noexcept;
	void handle_extern() noexcept;
	void handle_top_level() noexcept;
	std::unique_ptr<FunctionAST> parse_difinition() noexcept;
	std::unique_ptr<PrototypeAST> parse_prototype() noexcept;
	std::unique_ptr<FunctionAST> parse_top_level() noexcept;
	std::unique_ptr<ExprAST> parse_expression() noexcept;
	std::unique_ptr<ExprAST> parse_primary() noexcept;
	std::unique_ptr<ExprAST> parse_bin_op_RHS(int expr_prec, std::unique_ptr<ExprAST> LHS) noexcept;
	std::unique_ptr<ExprAST> parse_identifier_expr() noexcept;
	std::unique_ptr<NumberExprAST> parse_number_expr() noexcept;
	std::unique_ptr<ExprAST> parse_paron_expr() noexcept;
	// std::unique_ptr<PrototypeAST> parse_extern() noexcept;
};

}// namespace sail::llvm