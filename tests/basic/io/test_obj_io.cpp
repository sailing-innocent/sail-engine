#include "test_util.h"

#include "tiny_obj_loader.h"

namespace sail::test {

int test_load_obj() {
	tinyobj::attrib_t attrib;
	return 0;
}

}// namespace sail::test

TEST_SUITE("io") {
	TEST_CASE("tiny_obj") {
		CHECK(sail::test::test_load_obj() == 0);
	}
}
