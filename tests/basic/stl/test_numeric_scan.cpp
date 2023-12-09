#include "test_util.h"

#include <vector>
#include <numeric>
#include <functional>

TEST_SUITE("basic::numeric") {
	TEST_CASE("inclusive_scan_add") {
		std::vector data{3, 1, 4, 5, 9, 2, 6};
		std::vector gt{3, 4, 8, 13, 22, 24, 30};

		std::inclusive_scan(data.begin(), data.end(), data.begin());
		for (auto i = 0; i < data.size(); ++i) {
			REQUIRE(data[i] == gt[i]);
		}
	}

	TEST_CASE("exclusive_scan_add") {
		std::vector data{3, 1, 4, 5, 9, 2, 6};
		std::vector gt{0, 3, 4, 8, 13, 22, 24};

		std::exclusive_scan(data.begin(), data.end(), data.begin(), 0);
		for (auto i = 0; i < data.size(); ++i) {
			REQUIRE(data[i] == gt[i]);
		}
	}

	TEST_CASE("inclusive_scan_prod") {
		std::vector data{3, 1, 4, 5, 9, 2, 6};
		std::vector gt{3, 3, 12, 60, 540, 1080, 6480};

		std::inclusive_scan(data.begin(), data.end(), data.begin(), std::multiplies<>());
		for (auto i = 0; i < data.size(); ++i) {
			REQUIRE(data[i] == gt[i]);
		}
	}

	TEST_CASE("exclusive_scan_prod") {
		std::vector data{3, 1, 4, 5, 9, 2, 6};
		std::vector gt{1, 3, 3, 12, 60, 540, 1080};

		std::exclusive_scan(data.begin(), data.end(), data.begin(), 1, std::multiplies<>());
		for (auto i = 0; i < data.size(); ++i) {
			REQUIRE(data[i] == gt[i]);
		}
	}
}
