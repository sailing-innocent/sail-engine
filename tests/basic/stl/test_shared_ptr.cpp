#include "test_util.h"
#include <memory>

struct MyStruct {
	MyStruct() = default;
	MyStruct(int _idx)
		: index(_idx) {
	}
	int index = 0;
	int getIndex() { return index; }
};

TEST_SUITE("basic::memory") {
	TEST_CASE("shared_ptr") {
		MyStruct mystruct = MyStruct(1);
		REQUIRE(mystruct.getIndex() == 1);

		std::shared_ptr<MyStruct> p = std::make_shared<MyStruct>(mystruct);
		REQUIRE(p->getIndex() == 1);

		std::shared_ptr<MyStruct> np = p;
		REQUIRE(p->index == 1);
	}
}
