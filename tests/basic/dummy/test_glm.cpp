#include "test_util.h"

#include <glm/glm.hpp>
#define GLM_ENABLE_EXPERIMENTAL
#include <glm/gtx/quaternion.hpp>

namespace sail {

int test_glm() {
	// vector
	glm::vec4 v1(1.0f, 2.0f, 3.0f, 4.0f);
	// matrix
	glm::mat4 m1(1.0f);
	m1[0][0] = 1.0f;
	m1[1][0] = 2.0f;
	m1[1][1] = 2.0f;
	m1[2][2] = 3.0f;
	m1[3][3] = 4.0f;
	// glm is col major
	// 1, 2, 0, 0
	// 0, 2, 0, 0
	// 0, 0, 3, 0
	// 0, 0, 0, 4

	// matrix * vector
	glm::vec4 v2 = m1 * v1;

	// 1*1 + 2*2 + 0*3 + 0*4 = 5
	// 0*1 + 2*2 + 0*3 + 0*4 = 4
	// 0*1 + 0*2 + 3*3 + 0*4 = 9
	// 0*1 + 0*2 + 0*3 + 4*4 = 16
	CHECK(v2.x == doctest::Approx(5.0f));
	CHECK(v2.y == doctest::Approx(4.0f));
	CHECK(v2.z == doctest::Approx(9.0f));
	CHECK(v2.w == doctest::Approx(16.0f));

	glm::mat4 m2(1.0f);
	// 1, 2, 3, 4
	// 5, 6, 7, 8
	// 9, 10, 11, 12
	// 13, 14, 15, 16
	for (int i = 0; i < 4; i++) {
		for (int j = 0; j < 4; j++) {
			m2[i][j] = i + j * 4 + 1;
		}
	}

	// matrix * matrix
	glm::mat4 m3 = m1 * m2;
	// 1*1 + 2*5 + 0*9 + 0*13 = 11
	// 0*1 + 2*5 + 0*9 + 0*13 = 10
	// 0*1 + 0*5 + 3*9 + 0*13 = 27
	// 0*1 + 0*5 + 0*9 + 4*13 = 52
	// 1*2 + 2*6 + 0*10 + 0*14 = 14
	// 0*2 + 2*6 + 0*10 + 0*14 = 12
	// 0*2 + 0*6 + 3*10 + 0*14 = 30
	// 0*2 + 0*6 + 0*10 + 4*14 = 56
	// 1*3 + 2*7 + 0*11 + 0*15 = 17
	// 0*3 + 2*7 + 0*11 + 0*15 = 14
	// 0*3 + 0*7 + 3*11 + 0*15 = 33
	// 0*3 + 0*7 + 0*11 + 4*15 = 60
	// 1*4 + 2*8 + 0*12 + 0*16 = 20
	// 0*4 + 2*8 + 0*12 + 0*16 = 16
	// 0*4 + 0*8 + 3*12 + 0*16 = 36
	// 0*4 + 0*8 + 0*12 + 4*16 = 64

	CHECK(m3[0][0] == doctest::Approx(11.0f));
	CHECK(m3[0][1] == doctest::Approx(10.0f));
	CHECK(m3[0][2] == doctest::Approx(27.0f));
	CHECK(m3[0][3] == doctest::Approx(52.0f));
	CHECK(m3[1][0] == doctest::Approx(14.0f));
	CHECK(m3[1][1] == doctest::Approx(12.0f));
	CHECK(m3[1][2] == doctest::Approx(30.0f));
	CHECK(m3[1][3] == doctest::Approx(56.0f));
	CHECK(m3[2][0] == doctest::Approx(17.0f));
	CHECK(m3[2][1] == doctest::Approx(14.0f));
	CHECK(m3[2][2] == doctest::Approx(33.0f));
	CHECK(m3[2][3] == doctest::Approx(60.0f));
	CHECK(m3[3][0] == doctest::Approx(20.0f));
	CHECK(m3[3][1] == doctest::Approx(16.0f));
	CHECK(m3[3][2] == doctest::Approx(36.0f));
	CHECK(m3[3][3] == doctest::Approx(64.0f));

	// quaternion
	glm::quat q1(1.0f, 2.0f, 3.0f, 4.0f);
	// norm
	q1 = glm::normalize(q1);
	CHECK(q1.w == doctest::Approx(0.1825741858f));

	// to matrix
	glm::mat4 m4 = glm::mat4_cast(q1);
	// from euler
	glm::vec3 euler(1.0f, 2.0f, 3.0f);
	glm::quat q2 = glm::quat(euler);
	glm::mat4 m5 = glm::mat4_cast(q2);

	// quat * quat
	glm::quat q3 = q1 * q2;
	// matrix * matrix
	glm::mat4 m6 = m4 * m5;
	// check if they are the same
	glm::mat4 m7 = glm::mat4_cast(q3);

	for (int i = 0; i < 4; i++) {
		for (int j = 0; j < 4; j++) {
			CHECK(m6[i][j] == doctest::Approx(m7[i][j]));
		}
	}

	// from axis angle
	glm::vec3 axis(1.0f, 0.0f, 0.0f);
	glm::quat q4 = glm::angleAxis(3.14159265359f, axis);
	// from euler
	glm::vec3 euler2(3.14159265359f, 0.0f, 0.0f);
	glm::quat q5 = glm::quat(euler2);

	// check if they are the same
	CHECK(q4.w == doctest::Approx(q5.w));
	CHECK(q4.x == doctest::Approx(q5.x));
	CHECK(q4.y == doctest::Approx(q5.y));
	CHECK(q4.z == doctest::Approx(q5.z));

	return 0;
}

}// namespace sail

TEST_SUITE("basic::dummy") {
	TEST_CASE("test_glm") {
		CHECK(sail::test_glm() == 0);
	}
}