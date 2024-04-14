#include "test_util.h"

#include <memory>

namespace sail::test {

class Dummy {
public:
	int value = 1;
};

}// namespace sail::test

TEST_SUITE("basic::containers") {
	TEST_CASE("unique_ptr") {
		std::unique_ptr<sail::test::Dummy> ptr;
		ptr = std::make_unique<sail::test::Dummy>(std::move(sail::test::Dummy{}));
		auto ptr_moved = std::move(ptr);
		CHECK(ptr_moved->value == 1);
	}
}