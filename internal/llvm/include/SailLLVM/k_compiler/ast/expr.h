#pragma once
#include "SailLLVM/config.h"
#include <string>
#include <memory>
#include <vector>
#include <iostream>

namespace sail::llvm {

struct NumberExprAST;
struct VariableExprAST;
struct BinaryOpAST;
struct CallOpAST;
struct PrototypeAST;
struct FunctionAST;

struct SAIL_LLVM_API ExprVisitor {
	virtual void visit(NumberExprAST& node) = 0;
	virtual void visit(VariableExprAST& node) = 0;
	virtual void visit(BinaryOpAST& node) = 0;
	virtual void visit(CallOpAST& node) = 0;
	virtual void visit(PrototypeAST& node) = 0;
	virtual void visit(FunctionAST& node) = 0;
	virtual ~ExprVisitor() noexcept = default;
};
#define AST_COMMON() \
	void accept(ExprVisitor& visitor) override { visitor.visit(*this); }

class ExprAST {
public:
	virtual ~ExprAST() = default;
	virtual void accept(ExprVisitor& visitor) = 0;
};

class NumberExprAST final : public ExprAST {
	double m_val;

public:
	NumberExprAST(double val) : m_val(val) {
		std::cout << "Constructing Number AST " << m_val << "\n";
	}
	AST_COMMON()
};

class VariableExprAST final : public ExprAST {
	std::string m_name;

public:
	VariableExprAST(const std::string& name) : m_name(name) {
		std::cout << "Constructing Variable AST " << m_name << "\n";
	}
	AST_COMMON()
};

class BinaryOpAST final : public ExprAST {
	char m_op;
	std::unique_ptr<ExprAST> m_lhs, m_rhs;

public:
	BinaryOpAST(char op, std::unique_ptr<ExprAST> lhs, std::unique_ptr<ExprAST> rhs)
		: m_op(op), m_lhs(std::move(lhs)), m_rhs(std::move(rhs)) {
		std::cout << "Constructing Binary Op AST " << m_op << "\n";
	}
	AST_COMMON()
};

class CallOpAST final : public ExprAST {
	std::string m_callee;
	std::vector<std::unique_ptr<ExprAST>> m_args;

public:
	CallOpAST(const std::string& callee, std::vector<std::unique_ptr<ExprAST>> args)
		: m_callee(callee), m_args(std::move(args)) {
		std::cout << "Constructing CallOpAST\n";
	}
	AST_COMMON()
};

class PrototypeAST {
	std::string m_name;
	std::vector<std::string> m_args;

public:
	PrototypeAST(const std::string& name, std::vector<std::string> args)
		: m_name(name), m_args(std::move(args)) {
		std::cout << "Constructing Prototype AST: " << m_name << " with " << args.size() << " args\n";
	}

	const std::string& get_name() const { return m_name; }
};

class FunctionAST {
	std::unique_ptr<PrototypeAST> m_proto;
	std::unique_ptr<ExprAST> m_body;

public:
	FunctionAST(std::unique_ptr<PrototypeAST> proto, std::unique_ptr<ExprAST> body)
		: m_proto(std::move(proto)), m_body(std::move(body)) {
		std::cout << "Constructing Function AST \n";
	}
};

}// namespace sail::llvm