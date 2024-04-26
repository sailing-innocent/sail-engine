#include "SailLLVM/k_compiler/compiler.h"
#include "SailLLVM/k_compiler/ast/expr.h"
#include <iostream>
#include <cctype>
#include <memory>

namespace sail::llvm {

using namespace std;
Compiler::Compiler() {
	// supported bin op precedence
	m_bin_op_precedence['<'] = 10;
	m_bin_op_precedence['+'] = 20;
	m_bin_op_precedence['-'] = 30;
	m_bin_op_precedence['*'] = 40;
}
void Compiler::compile(string_view code) noexcept {
	m_code_size = code.size();
	// lexing & parsing
	m_code = code;
	get_next_tok();
	while (true) {
		switch (m_cur_tok) {
			case static_cast<int>(Token::TOK_EOF):
				return;
			case static_cast<int>(';'):// ignore top-level ;
				get_next_tok();
				break;
			case static_cast<int>(Token::TOK_DEF):
				handle_difinition();
				break;
			default:
				// Handle Top Level
				break;
		}
	}
}

int Compiler::get_next_tok() noexcept {
	m_cur_tok = get_tok(m_code);
	return m_cur_tok;
}
int Compiler::get_tok(string_view code) noexcept {
	// skip white space
	while (m_code_idx < m_code_size && isspace(code[m_code_idx])) {
		// std::cout << "isspace\n";
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
	int this_char = static_cast<int>(m_last_char);
	m_last_char = code[++m_code_idx];
	std::cout << "TOK_CHAR: ";
	std::cout << static_cast<char>(this_char) << '\n';
	return this_char;
}

void Compiler::handle_difinition() noexcept {
	auto func_ast = parse_difinition();
}

void Compiler::handle_top_level() noexcept {
	// ananomous function
}

void Compiler::handle_extern() noexcept {
	get_next_tok();	  // eat 'extern'
	parse_prototype();// only prototype
}

std::unique_ptr<FunctionAST> Compiler::parse_difinition() noexcept {
	get_next_tok();				   // eat 'def'
	auto proto = parse_prototype();// parse prototype
	if (!proto) { return nullptr; }

	if (auto e = parse_expression()) {
		return std::make_unique<FunctionAST>(std::move(proto), std::move(e));
	}
}

std::unique_ptr<PrototypeAST> Compiler::parse_prototype() noexcept {
	if (m_cur_tok != static_cast<int>(Token::TOK_IDENTIFIER)) {
		// error
		get_next_tok();
		return nullptr;
	}
	std::string fn_name = m_identifier_str;
	get_next_tok();
	if (m_cur_tok != static_cast<int>('(')) {
		// error
		get_next_tok();
		return nullptr;
	}
	std::vector<std::string> args_name;
	while (
		get_next_tok() ==
		static_cast<int>(Token::TOK_IDENTIFIER)) {
		args_name.push_back(m_identifier_str);
	}

	if (m_cur_tok != static_cast<int>(')')) {
		// error
		get_next_tok();
		return nullptr;
	}
	// success
	get_next_tok();// eat ')'
	return std::make_unique<PrototypeAST>(fn_name, std::move(args_name));
}

std::unique_ptr<ExprAST> Compiler::parse_expression() noexcept {
	auto LHS = parse_primary();
	if (!LHS) { return nullptr; }
	return parse_bin_op_RHS(0, std::move(LHS));
}

std::unique_ptr<ExprAST> Compiler::parse_primary() noexcept {
	// primtive parse
	switch (m_cur_tok) {
		default:
			return nullptr;// error
		case static_cast<int>(Token::TOK_IDENTIFIER):
			return parse_identifier_expr();
		case static_cast<int>(Token::TOK_NUMBER):
			return parse_number_expr();
		case static_cast<int>('('):
			return parse_paron_expr();
	}
}

std::unique_ptr<ExprAST> Compiler::parse_bin_op_RHS(
	int expr_prec,
	std::unique_ptr<ExprAST> LHS) noexcept {
	while (true) {
		int tok_prec = get_tok_prec();// get_tok_prec()
		if (tok_prec < expr_prec) {
			// -1, or just statement
			return LHS;
		}
		int bin_op = m_cur_tok;
		get_next_tok();// eat bin op

		auto RHS = parse_primary();
		if (!RHS) {
			return nullptr;
		}
		// if bin op bind tightly;
		// TODO
		LHS = std::make_unique<BinaryOpAST>(bin_op, std::move(LHS), std::move(RHS));
	}
}

int Compiler::get_tok_prec() noexcept {
	if (!isascii(m_cur_tok)) {
		// not ascii, eof, ...
		return -1;
	}
	int tok_prec = m_bin_op_precedence[m_cur_tok];
	if (tok_prec <= 0) {
		// not supported op
		return -1;
	}
	return tok_prec;
}

std::unique_ptr<ExprAST> Compiler::parse_identifier_expr() noexcept {
	std::string id_name = m_identifier_str;
	get_next_tok();

	if (m_cur_tok != static_cast<int>('(')) {
		// simple variable
		return std::make_unique<VariableExprAST>(id_name);
	}
	// else, call
	get_next_tok();// eat (
	std::vector<std::unique_ptr<ExprAST>> args;
	if (m_cur_tok != ')') {
		while (true) {
			if (auto arg = parse_expression()) {
				args.push_back(std::move(arg));
			} else {
				return nullptr;
			}

			if (m_cur_tok == ')') {
				break;
			}
			if (m_cur_tok != ',') {
				// error
				return nullptr;
			}
			get_next_tok();
		}
	}
	get_next_tok();// eat )
	return std::make_unique<CallOpAST>(id_name, std::move(args));
}

std::unique_ptr<NumberExprAST> Compiler::parse_number_expr() noexcept {
	auto result = std::make_unique<NumberExprAST>(m_num_val);
	get_next_tok();// consume the number
	return std::move(result);
}

std::unique_ptr<ExprAST> Compiler::parse_paron_expr() noexcept {
	// statement in (
	get_next_tok();
	auto v = parse_expression();
	if (!v) { return nullptr; }
	if (m_cur_tok != ')') {
		// error
		get_next_tok();
		return nullptr;
	}
	get_next_tok();
	return v;
}

}// namespace sail::llvm