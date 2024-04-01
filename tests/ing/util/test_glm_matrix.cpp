#include "test_util.h"

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

TEST_SUITE("ing::util") {
	TEST_CASE("glm_mt") {
		// clang-format off
		glm::mat3 R = {1.0f, 0.0f, 1.0f, // 1st col
					   0.0f, 1.0f, 1.0f, // 2nd col
					   0.0f, 0.0f, 1.0f  // 3rd col
					};
		// clang-format on
		// retrival
		// [i][j] means i-th col, j-th row
		CHECK(R[0][0] == doctest::Approx(1.0f));
		CHECK(R[0][1] == doctest::Approx(0.0f));
		CHECK(R[0][2] == doctest::Approx(1.0f));
		CHECK(R[1][0] == doctest::Approx(0.0f));
		CHECK(R[1][1] == doctest::Approx(1.0f));
		CHECK(R[1][2] == doctest::Approx(1.0f));
		CHECK(R[2][0] == doctest::Approx(0.0f));
		CHECK(R[2][1] == doctest::Approx(0.0f));
		CHECK(R[2][2] == doctest::Approx(1.0f));

		glm::mat3 RT = glm::transpose(R);
		CHECK(RT[0][0] == doctest::Approx(1.0f));
		CHECK(RT[0][1] == doctest::Approx(0.0f));
		CHECK(RT[0][2] == doctest::Approx(0.0f));
		CHECK(RT[1][0] == doctest::Approx(0.0f));
		CHECK(RT[1][1] == doctest::Approx(1.0f));
		CHECK(RT[1][2] == doctest::Approx(0.0f));
		CHECK(RT[2][0] == doctest::Approx(1.0f));
		CHECK(RT[2][1] == doctest::Approx(1.0f));
		CHECK(RT[2][2] == doctest::Approx(1.0f));
	}
	TEST_CASE("glm_mv") {
		// GLM is col-major
		// R =
		// 1, 0, 0
		// 0, 1, 0
		// 1, 1, 1
		// v = [1, 2, 3]^T
		// Rv
		// clang-format off
		glm::mat3 R = {1.0f, 0.0f, 1.0f, // 1st col
					   0.0f, 1.0f, 1.0f, // 2nd col
					   0.0f, 0.0f, 1.0f  // 3rd col
					};
		// clang-format on
		// retrival
		// [i][j] means i-th col, j-th row

		glm::vec3 v = {1.0f, 2.0f, 3.0f};
		auto rv = R * v;
		CHECK(rv.x == doctest::Approx(1.0f));
		CHECK(rv.y == doctest::Approx(2.0f));
		CHECK(rv.z == doctest::Approx(6.0f));
	}

	TEST_CASE("glm_mm") {
		// GLM is col-major
		// A =
		// 1, 0, 0
		// 0, 1, 0
		// 1, 1, 1
		// B =
		// 1, 1, 0
		// 0, 1, 1
		// 0, 0, 1
		// AB
		// 1 1 0
		// 0 1 1
		// 1 2 2
		// clang-format off
		glm::mat3 A = {1.0f, 0.0f, 1.0f, // 1st col
					   0.0f, 1.0f, 1.0f, // 2nd col
					   0.0f, 0.0f, 1.0f  // 3rd col
					};
		glm::mat3 B = {1.0f, 0.0f, 0.0f, // 1st col
					   1.0f, 1.0f, 0.0f, // 2nd col
					   0.0f, 1.0f, 1.0f  // 3rd col
					};
		// clang-format on
		glm::mat3 AB = A * B;

		CHECK(AB[0][0] == doctest::Approx(1.0f));
		CHECK(AB[0][1] == doctest::Approx(0.0f));
		CHECK(AB[0][2] == doctest::Approx(1.0f));
		CHECK(AB[1][0] == doctest::Approx(1.0f));
		CHECK(AB[1][1] == doctest::Approx(1.0f));
		CHECK(AB[1][2] == doctest::Approx(2.0f));
		CHECK(AB[2][0] == doctest::Approx(0.0f));
		CHECK(AB[2][1] == doctest::Approx(1.0f));
		CHECK(AB[2][2] == doctest::Approx(2.0f));
	}
}