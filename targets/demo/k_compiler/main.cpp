#include <cstdio>
#include <string>
#include <vector>
#include <memory>

namespace sail::llvm {

enum struct Token : int {
	TOK_EOF = -1,
	TOK_DEF = -2,
	TOK_EXTERN = -3,
	TOK_IDENTIFIER = -4,
	TOK_NUMBER = -5
};

static std::string IdentifierStr;// filled if TOK_IDENTIFIER
static double NumVal;			 // filled if TOK_NUMBER

static int gettok() {
	static int LastChar = ' ';
	// skip whitespaces
	while (isspace(LastChar)) {
		LastChar = getchar();
	}
	// if is alpha
	if (isalpha(LastChar)) {
		// read str
		IdentifierStr = LastChar;
		while (isalnum((LastChar = getchar()))) {
			IdentifierStr += LastChar;
		}
		// switch if is keyword
		if (IdentifierStr == "def") {
			return static_cast<int>(Token::TOK_DEF);
		}
		if (IdentifierStr == "extern") {
			return static_cast<int>(Token::TOK_EXTERN);
		}
		// else is identifier
		return static_cast<int>(Token::TOK_IDENTIFIER);
	}

	// if is digit
	if (isdigit(LastChar) || LastChar == '.') {
		std::string NumStr;
		do {
			NumStr += LastChar;
			LastChar = getchar();
		} while (isdigit(LastChar) || LastChar == '.');
		NumVal = strtod(NumStr.c_str(), nullptr);
		return static_cast<int>(Token::TOK_NUMBER);
	}

	// if is comment
	if (LastChar == '#') {
		do {
			LastChar = getchar();
		} while (LastChar != EOF && LastChar != '\n' && LastChar != '\r');
		if (LastChar != EOF) {
			return gettok();
		}
	}
	// if EOF
	if (LastChar == EOF) {
		return static_cast<int>(Token::TOK_EOF);
	}

	int ThisChar = LastChar;
	LastChar = getchar();
	return ThisChar;
}

class ExprAST {
public:
	virtual ~ExprAST() = default;
};

class NumberExprAST : public ExprAST {
public:
	NumberExprAST(double Val) : Val(Val) {}

private:
	double Val;
};

class VariableExprAST : public ExprAST {
public:
	VariableExprAST(const std::string& Name) : Name(Name) {}

private:
	std::string Name;
};

class BinaryExprAST : public ExprAST {
public:
	BinaryExprAST(char Op, std::unique_ptr<ExprAST> LHS, std::unique_ptr<ExprAST> RHS)
		: Op(Op), LHS(std::move(LHS)), RHS(std::move(RHS)) {}

private:
	char Op;
	std::unique_ptr<ExprAST> LHS, RHS;
};

class CallExprAST : public ExprAST {
public:
	CallExprAST(const std::string& Callee, std::vector<std::unique_ptr<ExprAST>> Args)
		: Callee(Callee), Args(std::move(Args)) {}

private:
	std::string Callee;
	std::vector<std::unique_ptr<ExprAST>> Args;
};

class PrototypeAST {
public:
	PrototypeAST(const std::string& Name, std::vector<std::string> Args)
		: Name(Name), Args(std::move(Args)) {}
	const std::string& getName() const { return Name; }

private:
	std::string Name;
	std::vector<std::string> Args;
};

class FunctionAST {
	std::unique_ptr<PrototypeAST> Proto;
	std::unique_ptr<ExprAST> Body;

public:
	FunctionAST(std::unique_ptr<PrototypeAST> Proto, std::unique_ptr<ExprAST> Body)
		: Proto(std::move(Proto)), Body(std::move(Body)) {}
};

static int CurTok;
static int getNextTok() { return CurTok = gettok(); }

static std::unique_ptr<ExprAST> ParseNumberExpr() {
	auto Result = std::make_unique<NumberExprAST>(NumVal);
	getNextTok();
	return Result;
}
// driver
static void MainLoop() {
	while (true) {
		fprintf(stderr, "ready>");
		switch (CurTok) {
			case static_cast<int>(Token::TOK_EOF):
				return;
			default:
				// top level expression
				return;
		}
	}
}

}// namespace sail::llvm

using namespace sail::llvm;
int main() {
	return 0;
}