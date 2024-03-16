#include "test_util.h"

#include <vector>
#include <list>
#include <deque>
#include <algorithm>

TEST_SUITE("basic::algorithm") {
	TEST_CASE("iterator") {
		const int array_size = 7;
		int ia[array_size] = {0, 1, 2, 3, 4, 5, 6};
		std::vector<int> ivect{ia, ia + array_size};

		std::vector<int>::iterator it1 = std::find(ivect.begin(), ivect.end(), 4);
		REQUIRE(it1 == ivect.begin() + 4);

		std::list<int> ilist(ia, ia + array_size);
		std::list<int>::iterator it2 = std::find(ilist.begin(), ilist.end(), 8);
		REQUIRE(it2 == ilist.end());// not support +

		std::deque<int> ideque(ia, ia + array_size);
		std::deque<int>::iterator it3 = std::find(ideque.begin(), ideque.end(), 4);
		REQUIRE(it3 == ideque.begin() + 4);
	}
}
