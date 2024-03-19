// leetcode 079, word exists, medium
// 2023-10-28
// NOT DONE YET

#include "test_util.h"
#include <vector>
#include <string>
#include <iostream>

namespace sail::test {

void print_path(std::vector<std::vector<int>>& path, std::vector<std::vector<char>>& board) {
	std::cout << "Path: \n";
	for (auto ipath = 0; ipath < path.size(); ipath++) {
		std::cout << "Path " << ipath << ": ";
		for (auto jpath = 0; jpath < path[ipath].size() / 2; jpath++) {
			int curr_i = path[ipath][2 * jpath + 0];
			int curr_j = path[ipath][2 * jpath + 1];
			std::cout << "(" << curr_i << "," << curr_j
					  << "): " << board[curr_i][curr_j] << ">>";
		}
		std::cout << "\n";
	}
	std::cout << std::endl;
}

bool word_exists(std::vector<std::vector<char>>& board, std::string word) {
	// TODO: visited point is not allowed
	// path[i][2*j+0], path[i][2*j+1] = board index
	bool able = true;
	std::vector<std::vector<int>> path;

	int w = board[0].size();
	int h = board.size();

	while (able) {
		if (path.size() < 1) {
			// first search
			for (int i = 0; i < h; i++) {
				for (int j = 0; j < w; j++) {
					if (board[i][j] == word[0]) {
						path.push_back({i, j});
					}
				}
			}
		}
		if (path.size() < 1)// no starting point
		{
			return false;
		}
		able = false;
		int max_path = path.size();

		for (int ipath = 0; ipath < max_path; ipath++) {
			int curr_len = path[ipath].size() / 2;
			if (curr_len == word.size()) {
				return true;
			}
			int curr_i = path[ipath][2 * curr_len - 2];
			int curr_j = path[ipath][2 * curr_len - 1];
			bool found = false;
			for (auto i = -1; i < 2; i++) {
				for (auto j = -1; j < 2; j++) {
					if (i == 0 && j == 0) {
						continue;
					}
					int idx_i = curr_i + i;
					int idx_j = curr_j + j;
					if (idx_i < 0 || idx_i >= h || idx_j < 0 || idx_j >= w) {
						continue;
					}
					if (board[idx_i][idx_j] == word[curr_len]) {
						if (found) {
							// copy the path
							std::vector<int> new_path;
							for (auto k = 0; k < path[ipath].size(); k++) {
								new_path.push_back(path[ipath][k]);
							}
							new_path.push_back(idx_i);
							new_path.push_back(idx_j);
							path.push_back(new_path);
						} else {
							// first found next
							path[ipath].push_back(idx_i);
							path[ipath].push_back(idx_j);
							found = true;
						}
					}
				}
			}
			able = able || found;
			print_path(path, board);
		}
	}
	return false;
}

}// namespace sail::test

TEST_CASE("lc_079") {
	using namespace sail::test;
	std::vector<std::vector<char>> board = {
		{'A', 'B', 'C', 'E'}, {'S', 'F', 'C', 'S'}, {'A', 'D', 'E', 'E'}};
	std::string word = "ABCCED";
	REQUIRE(word_exists(board, word) == true);
	word = "SEE";
	REQUIRE(word_exists(board, word) == true);
	word = "ABCB";
	REQUIRE(word_exists(board, word) == false);
}