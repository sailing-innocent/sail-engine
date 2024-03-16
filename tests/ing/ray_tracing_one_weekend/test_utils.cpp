#include "test_util.h"
#include "vec3.h"
#include "ray.h"
#include "rtow.h"
#include <iostream>

namespace ing::test {

int test_vec3() {
	ing::rtow::vec3 v{0.0, 1.0, 2.0};
	CHECK(v[0] == 0.0);
	CHECK(v[1] == 1.0);
	CHECK(v[2] == 2.0);
	return 0;
}

int test_ray() {
	ing::rtow::point3 origin{0.0, 0.0, 0.0};
	ing::rtow::vec3 direction{1.0, 1.0, 1.0};
	ing::rtow::ray r{origin, direction};
	CHECK(r.at(0.0) == origin);
	CHECK(r.at(1.0) == origin + direction);
	CHECK(r.at(0.5) == ing::rtow::point3{0.5, 0.5, 0.5});
	return 0;
}

int test_refract() {
	ing::rtow::vec3 uv{1.0, 1.0, 0.0};
	ing::rtow::vec3 n{0.0, -1.0, 0.0};
	auto etai_over_etat = 1.5;
	auto refracted = ing::rtow::refract(uv, n, etai_over_etat);
	std::cout << "refracted: " << refracted[0] << " " << refracted[1] << " " << refracted[2] << " " << std::endl;

	double cos_theta = dot(-uv, n);
	double cos_theta_prime = dot(refracted, n);
	double sin_theta = std::sqrt(1.0 - cos_theta * cos_theta);
	double sin_theta_prime = std::sqrt(1.0 - cos_theta_prime * cos_theta_prime);
	CHECK(sin_theta / sin_theta_prime == etai_over_etat);

	return 0;
}

}// namespace ing::test

TEST_CASE("rtow_01") {
	REQUIRE(ing::test::test_vec3() == 0);
	REQUIRE(ing::test::test_ray() == 0);
	REQUIRE(ing::test::test_refract() == 0);
}