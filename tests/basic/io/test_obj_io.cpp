#include "test_util.h"

#include "tiny_obj_loader.h"
#include <string>
#include <vector>

namespace sail::test {

int test_load_obj() {
	tinyobj::attrib_t attrib;
	std::vector<tinyobj::shape_t> shapes;
	std::vector<tinyobj::material_t> materials;// TODO: use materials
	std::string warn, err;
	if (!tinyobj::LoadObj(&attrib, &shapes, &materials, &warn, &err, "assets/models/triangle.obj")) {
		return 1;
	}
	size_t vertex_count = attrib.GetVertices().size();
	CHECK(vertex_count == 9);
	CHECK(shapes.size() == 1);
	return 0;
}

}// namespace sail::test

TEST_SUITE("io") {
	TEST_CASE("tiny_obj") {
		CHECK(sail::test::test_load_obj() == 0);
	}
}
