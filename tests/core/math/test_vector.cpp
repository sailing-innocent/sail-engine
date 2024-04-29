#include "test_util.h"
#include "SailMath/type/vector.hpp"

namespace sail::test {

template<typename T, size_t N>
int test_vector() {
	Vector<T, N> v;
	return 0;
}

}// namespace sail::test

using namespace sail::test;
TEST_SUITE("core::math") {
	TEST_CASE("vector") {
		CHECK(test_vector<float, 3>() == 0);
	}
}