#pragma once
#include "SailLLVM/config.h"
#include <memory>
#include <string>

namespace sail::llvm {

// A State Machine Compiler
struct SAIL_LLVM_API Compiler {
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

	void compile(std::string_view code) noexcept;
	char m_last_char = ' ';// for get_tok
	[[nodiscard]] int get_tok(std::string_view code) noexcept;
};

}// namespace sail::llvm