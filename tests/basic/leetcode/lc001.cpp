/**
 * @file lc001.cpp
 * @author sailing-innocent
 * @date 2023-02-18
 * @brief the leetcode 001 two sum
*/

#include "test_util.h"

#include <vector>
#include <span>
#include <array>
#include <unordered_map>

namespace sail::test {

void two_sum_brute_force(std::span<int> nums, int target, std::span<int> output) {
	size_t len = nums.size();
	for (auto i = 0; i < len - 1; i++) {
		for (auto j = 1; j < len; j++) {
			if (nums[i] + nums[j] == target) {
				output[0] = i;
				output[1] = j;
			}
		}
	}
}

void two_sum_hash_code(std::span<int> nums, int target, std::span<int, 2> output) {
	size_t len = nums.size();
	std::unordered_map<int, int> hash_map;
	int i = 0;
	while (i < len) {
		auto it = hash_map.find(target - nums[i]);
		if (it != hash_map.end()) {
			// found!
			output[0] = it->second;
			output[1] = i;
			return;// early return
		}
		hash_map[nums[i]] = i;
		i = i + 1;
	}
}

std::array<int, 2> two_sum(std::span<int> nums, int target, int method = 0) {
	std::array<int, 2> output = {-1, -1};
	switch (method) {
		case 0:
			two_sum_brute_force(nums, target, output);
			break;
		case 1:
			two_sum_hash_code(nums, target, output);
		default:
			break;
	}
	return output;
}

}//namespace sail::test

TEST_CASE("lc_001") {
	using namespace sail::test;
	std::vector<int> nums = {3, 2, 4};
	int target = 6;
	std::array<int, 2> expected = {1, 2};
	auto output = two_sum(nums, target);
	REQUIRE(output[0] == expected[0]);
	REQUIRE(output[1] == expected[1]);
	output = two_sum(nums, target, 1);
	REQUIRE(output[0] == expected[0]);
	REQUIRE(output[1] == expected[1]);
}
