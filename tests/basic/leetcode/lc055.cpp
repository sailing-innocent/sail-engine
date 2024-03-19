// leetcode 055. Jump Game
// 2023-10-28

#include "test_util.h"
#include <vector>
#include <span>

namespace sail::test {

bool test_jump_game(std::span<int> nums) {
	int prev_max = 0;
	int curr_max = 0;
	int i = 0;
	while (i < nums.size()) {
		if (prev_max < i) {
			return false;
		}
		curr_max = nums[i] + i;
		if (prev_max < curr_max) {
			prev_max = curr_max;
		}
		i++;
	}
	return true;
}

}// namespace sail::test

TEST_CASE("lc_055") {
	using namespace sail::test;
	std::vector<int> nums = {2, 3, 1, 1, 4};
	REQUIRE(test_jump_game(nums) == true);
	nums = {3, 2, 1, 0, 4};
	REQUIRE(test_jump_game(nums) == false);
}