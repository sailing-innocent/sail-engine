#include "test_util.h"

namespace sail::test {

struct DummyData {
	int dt;
};

struct DummyToMove {
	DummyToMove(int dt) {
		this->foo = new DummyData();
		this->foo->dt = dt;
	}
	DummyToMove(const DummyToMove& d) {
		this->foo = new DummyData();
		this->foo->dt = d.foo->dt;
	}
	DummyToMove(DummyToMove&& d) {
		this->foo = d.foo;
	}
	DummyData* foo;
};

int test_move() {
	DummyToMove a{1};
	DummyToMove a_copy = a;
	DummyToMove a_move = std::move(a);
	CHECK(a_copy.foo->dt == 1);
	CHECK(a_move.foo->dt == 1);
	// del a data
	delete[] a.foo;
	// copy data is not deleted
	CHECK(a_copy.foo->dt == 1);
	// move data is deleted
	// ub
	CHECK_FALSE(a_move.foo->dt == 1);

	return 0;
}

}// namespace sail::test

TEST_SUITE("basic::stl") {
	TEST_CASE("test_move") {
		REQUIRE(sail::test::test_move() == 0);
	}
}