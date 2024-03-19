/**
 * @file lc152.cpp
 * @author sailing-innocent
 * @date 2023-09-19
 * @note the leetcode 152: maximum product sub array
*/
#include "test_util.h"
#include <vector>

namespace sail::test {

int max_prod_iter(std::vector<int>& nums, size_t arr_start, size_t arr_end) {
	size_t left_minus_pos = arr_start;
	size_t right_minus_pos = arr_end;

	if (arr_start == arr_end) {
		return nums[arr_start];
	}
	while (arr_start < arr_end) {
		int minus_count = 0;
		for (auto i = arr_start; i <= arr_end; i++) {
			if (nums[i] == 0) {
				if (arr_start == i) {
					return std::max(max_prod_iter(nums, arr_start + 1, arr_end), 0);
				}
				if (arr_end == i) {
					return std::max(max_prod_iter(nums, arr_start, arr_end - 1), 0);
				}
				auto max_left = max_prod_iter(nums, arr_start, i - 1);
				auto max_right = max_prod_iter(nums, i + 1, arr_end);
				return std::max(max_left, std::max(max_right, 0));
			}
			if (nums[i] < 0) {
				minus_count++;
				if (minus_count == 1) {
					left_minus_pos = i;
				}
				right_minus_pos = i;
			}
		};
		if (minus_count % 2 == 0) {
			// quolified
			break;
		} else {
			auto left_max_left = nums[arr_start];
			if (left_minus_pos > arr_start) {
				left_max_left = max_prod_iter(nums, arr_start, left_minus_pos - 1);
			}
			auto left_max_right = left_max_left;
			if (left_minus_pos < arr_end) {
				left_max_right = max_prod_iter(nums, left_minus_pos + 1, arr_end);
			}

			auto right_max_right = nums[arr_end];
			if (right_minus_pos < arr_end) {
				max_prod_iter(nums, right_minus_pos + 1, arr_end);
			}
			auto right_max_left = right_max_right;

			if (right_minus_pos > arr_start) {
				right_max_left = max_prod_iter(nums, arr_start, right_minus_pos - 1);
			}
			return std::max(std::max(left_max_left, left_max_right),
							std::max(right_max_left, right_max_right));
		}
	}
	// this is the qualified array
	int output = 1;
	for (auto i = arr_start; i <= arr_end; i++) {
		output *= nums[i];
	}
	return output;
}

int max_prod_subarray(std::vector<int>& nums) {
	size_t len = nums.size();
	return max_prod_iter(nums, 0, len - 1);
}

}// namespace sail::test

TEST_CASE("lc_152") {
	using namespace sail::test;
	SUBCASE("testcase_01") {
		std::vector<int> nums = {2, 3, -2, 4};
		REQUIRE(max_prod_subarray(nums) == 6);
	}
	SUBCASE("testcase_02") {
		std::vector<int> nums = {-2, 0, -1};
		REQUIRE(max_prod_subarray(nums) == 0);
	}
	SUBCASE("testcase_03") {
		std::vector<int> nums = {-2, 3, -4};
		REQUIRE(max_prod_subarray(nums) == 24);
	}
	SUBCASE("testcase_04") {
		std::vector<int> nums = {-2, 3, -4, 0, 2, 3, -2, 4};
		REQUIRE(max_prod_subarray(nums) == 24);
	}
	SUBCASE("testcase_05") {
		std::vector<int> nums = {-3, -1, -1};
		REQUIRE(max_prod_subarray(nums) == 3);
	}
	SUBCASE("testcase_06") {
		std::vector<int> nums = {-1, -1, -3};
		REQUIRE(max_prod_subarray(nums) == 3);
	}
	SUBCASE("testcase_07") {
		std::vector<int> nums = {0, 2};
		REQUIRE(max_prod_subarray(nums) == 2);
	}
	SUBCASE("testcase_08") {
		std::vector<int> nums = {1, -2, 3, -4, -3, -4, -3};
		REQUIRE(max_prod_subarray(nums) == 432);
	}
	SUBCASE("testcase_09") {
		std::vector<int> nums = {-1, 1, 2, 1};
		REQUIRE(max_prod_subarray(nums) == 2);
	}
}