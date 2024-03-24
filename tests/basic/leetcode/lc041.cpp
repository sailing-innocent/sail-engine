#include "test_util.h"

#include <vector>

// 这道题边界情况特别多，高血压慎入

namespace sail::test {
int first_missing_positive(std::vector<int>& nums) {
	if (nums.size() <= 1) {
		if (nums[0] == 1) {
			return 2;
		}
		return 1;
	}
	for (auto i = 0; i < nums.size(); i++) {
		while (nums[i] >= 1 && nums[i] < nums.size() + 1 && nums[nums[i] - 1] != nums[i]) {
			int temp = nums[nums[i] - 1];
			nums[nums[i] - 1] = nums[i];
			nums[i] = temp;
		}
	}
	int res = 1;
	while (res - 1 < nums.size() && nums[res - 1] == res) {
		res = res + 1;
	}
	return res;
}

}// namespace sail::test

TEST_CASE("lc_041") {
	using namespace sail::test;
	std::vector<int> s1{3, 4, -1, 1};
	REQUIRE(first_missing_positive(s1) == 2);
	std::vector<int> s2{2, 1};
	REQUIRE(first_missing_positive(s2) == 3);
	std::vector<int> s3{1, 2, 2, 1, 3, 1, 0, 4, 0};
	REQUIRE(first_missing_positive(s3) == 5);
	std::vector<int> s4{2147483647, 2147483646, 2147483645, 3, 2, 1, -1, 0, -2147483647};
	REQUIRE(first_missing_positive(s4) == 4);
}
