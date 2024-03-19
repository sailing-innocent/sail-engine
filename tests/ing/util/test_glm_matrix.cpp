#include "test_util.h"

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

TEST_SUITE("ing::util") {
	TEST_CASE("glm_mv") {
		// GLM is col-major
		// R =
		// 1, 0, 0
		// 0, 1, 0
		// 1, 1, 1
		// v = [1, 2, 3]^T
		// Rv
		glm::mat3 R = {1.0f,
					   0.0f,
					   1.0f,
					   0.0f,
					   1.0f,
					   1.0f,
					   0.0f,
					   0.0f,
					   1.0f};
		glm::vec3 v = {1.0f, 2.0f, 3.0f};
		auto rv = R * v;
		CHECK(rv.x == doctest::Approx(1.0f));
		CHECK(rv.y == doctest::Approx(2.0f));
		CHECK(rv.z == doctest::Approx(6.0f));
	}
}