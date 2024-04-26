#include "SailLLVM/k_compiler/compiler.h"
#include <iostream>
#include <cctype>

namespace sail::llvm {

using namespace std;

void Compiler::compile(string_view code) noexcept {
	m_code_size = code.size();
	while (m_code_idx < m_code_size) {
		int tok = get_tok(code);
	}
}

int Compiler::get_tok(string_view code) noexcept {
	// skip white space
	while (m_code_idx < m_code_size && isspace(code[m_code_idx])) {
		std::cout << "isspace\n";
		++m_code_idx;
	}
	m_last_char = code[m_code_idx];
	if (isalpha(m_last_char)) {// Identifier [a-zA-Z][a-zA-Z0-9]*
		m_identifier_str = m_last_char;
		// accumulate identifier string
		while (++m_code_idx < m_code_size) {
			m_last_char = code[m_code_idx];
			if (!isalnum(m_last_char)) {
				break;
			}
			m_identifier_str += code[m_code_idx];
		}
		// check if it is a keyword
		if (m_identifier_str == "def") {
			std::cout << "TOK_DEF\n";
			return static_cast<int>(Token::TOK_DEF);
		}
		if (m_identifier_str == "extern") {
			std::cout << "TOK_EXTERN\n";
			return static_cast<int>(Token::TOK_EXTERN);
		}
		// not a keyword
		std::cout << "TOK_IDENTIFIER: ";
		std::cout << m_identifier_str << '\n';
		return static_cast<int>(Token::TOK_IDENTIFIER);
	}
	if (isdigit(m_last_char) || m_last_char == '.') {// Number [0-9.]+
		// incorrect for 1.2.3
		string num_str;
		do {
			num_str += m_last_char;
			m_last_char = code[++m_code_idx];
		} while (m_code_idx < m_code_size && (isdigit(m_last_char) || m_last_char == '.'));
		m_num_val = strtod(num_str.c_str(), nullptr);
		std::cout << "TOK_NUMBER: ";
		std::cout << m_num_val << '\n';
		return static_cast<int>(Token::TOK_NUMBER);
	}

	// Comment
	if (m_last_char == '#') {
		// skip to end of line
		while (++m_code_idx < m_code_size && code[m_code_idx] != '\n' && code[m_code_idx] != '\r') {
		}
		if (m_code_idx < m_code_size) {
			return get_tok(code);
		}
	}
	// EOF
	if (m_code_idx == m_code_size) {
		std::cout << "TOK_EOF\n";
		return static_cast<int>(Token::TOK_EOF);
	}
	// single ascii char
	char this_char = m_last_char;
	m_last_char = code[++m_code_idx];
	std::cout << "TOK_CHAR: ";
	std::cout << this_char << '\n';
	return this_char;
}

}// namespace sail::llvm