#include "test_util.h"

#include <string>
#include <vector>

namespace sail::test {

std::vector<std::string> get_way(int x, int y, int n) {
	std::vector<std::string> res{};
	if (n == x && x == y) {
		res.push_back("");
	}
	if (n > x) {
		std::vector<std::string> downRes = get_way(x + 1, y, n);
		for (auto item : downRes) {
			item = "(" + item;
			res.push_back(item);
		}
	}
	if (x > y) {
		std::vector<std::string> rightRes = get_way(x, y + 1, n);
		for (auto item : rightRes) {
			item = ")" + item;
			res.push_back(item);
		}
	}
	return res;
}
std::vector<std::string> gen_pair(int n) {
	return get_way(0, 0, n);
}

}// namespace sail::test

TEST_CASE("lc_022") {
	using namespace sail::test;
	std::vector<std::string> res = gen_pair(3);
	REQUIRE(res.size() == 5);
	REQUIRE(res[0] == "((()))");
	REQUIRE(res[1] == "(()())");
	REQUIRE(res[2] == "(())()");
	REQUIRE(res[3] == "()(())");
	REQUIRE(res[4] == "()()()");
}