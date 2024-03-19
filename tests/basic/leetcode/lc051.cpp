/**
 * @file lc051.cpp
 * @author sailing-innocent
 * @date 2024-03-07 [NOT DONE]
 * @brief lc051 N-Queens
*/

#include "test_util.h"

// input 4
// output: all distinct solution for N-Queens
// ["Q...", ".Q..", "..Q.", "...Q"] ...
// 似乎斜着也不行，事情开始变得好玩了起来

#include <iostream>
#include <vector>
#include <string>

namespace sail::test {

void print_n_queens(std::vector<std::vector<std::string>>& res) {
	std::cout << "[" << std::endl;
	for (auto& sol : res) {
		std::cout << "[" << std::endl;
		for (auto& row : sol) {
			std::cout << row << std::endl;
		}
		std::cout << "]" << std::endl;
	}
	std::cout << "]" << std::endl;
}

std::vector<std::vector<std::string>> solve_n_queen(int n) {
	std::vector<std::vector<std::string>> res;
	std::vector<std::string> res_item;
	if (n == 1) {
		res_item.push_back("Q");
		res.push_back(res_item);
		return res;
	}
	std::vector<std::vector<std::string>> last_res = solve_n_queen(n - 1);

	for (auto& item : last_res) {
		for (auto i = 0; i < n; i++) {
			// i-rh col
			auto item_copy = item;
			std::string new_row(n, '.');
			// append Q
			new_row[i] = 'Q';
			for (auto& row : item_copy) {
				// insert '.' to i-th col
				row.insert(i, ".");
			}
			item_copy.push_back(new_row);
			res.push_back(item_copy);
		}
	}
	return res;
}

int test_n_queens() {
	auto res = solve_n_queen(2);
	print_n_queens(res);
	return 0;
}

}// namespace sail::test

TEST_CASE("lc_051") {
	REQUIRE(sail::test::test_n_queens() == 0);
}