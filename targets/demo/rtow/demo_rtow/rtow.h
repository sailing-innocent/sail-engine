#pragma once

// the common header for ray-tracing-one-weekend

#include <cmath>
#include <limits>
#include <random>

#include "vec3.h"
#include <fstream>
#include <filesystem>

// Constants

namespace sail::rtow {

const double infinity = std::numeric_limits<double>::infinity();
const double pi = 3.1415926535897932385;

// utility functions
inline double degrees_to_radians_double(double degrees) {
	return degrees * pi / 180.0;
}

inline double random_double() {
	// return random real in [0,1]
	static std::uniform_real_distribution<double> distribution(0.0, 1.0);
	static std::mt19937 generator;
	return distribution(generator);
}

inline double random_double(double d_min, double d_max) {
	// return random real in [d_min,d_max]
	return d_min + (d_max - d_min) * random_double();
}

inline vec3 random_vec3() {
	vec3 v{random_double(), random_double(), random_double()};
	return v;
}

inline vec3 random_vec3(double d_min, double d_max) {
	return vec3{random_double(d_min, d_max), random_double(d_min, d_max), random_double(d_min, d_max)};
}

inline vec3 random_in_unit_sphere() {
	// return a random point in the unit sphere
	while (true) {
		auto p = random_vec3(-1, 1);
		if (dot(p, p) >= 1) {
			continue;
		}
		return p;
	}
}

inline vec3 random_unit_vector() {
	// return a random unit vector
	return unit_vector(random_in_unit_sphere());
}

inline vec3 random_on_hemisphere(const vec3& normal) {
	// return a random point on the hemisphere
	auto p = random_in_unit_sphere();
	return dot(p, normal) > 0 ? p : -p;
}

inline void write_image(std::filesystem::path& of_path, std::vector<char>& image_buffer, int image_width, int image_height) {
	std::ofstream ofs;
	ofs.open(of_path, std::ios::binary);
	ofs << "P6\n"
		<< image_width << " " << image_height << "\n255\n";
	ofs.write(image_buffer.data(), image_width * image_height * 3);
	ofs.close();
}

inline double linear_to_gamma(double linear) {
	return std::pow(linear, 1.0 / 2.2);
}

inline bool near_zero(vec3& v) {
	auto s = 1e-8;
	return (std::fabs(v[0]) < s) && (std::fabs(v[1]) < s) && (std::fabs(v[2]) < s);
}

inline vec3 reflect(const vec3& v, const vec3& n) {
	return v - 2 * dot(v, n) * n;
}

inline vec3 refract(const vec3& uv, const vec3& n, double etai_over_etat) {
	auto cos_theta = fmin(dot(-uv, n), 1.0);
	vec3 r_out_perp = etai_over_etat * (uv + cos_theta * n);
	vec3 r_out_parallel = -sqrt(fabs(1.0 - dot(r_out_perp, r_out_perp))) * n;
	return r_out_perp + r_out_parallel;
}

}// namespace sail::rtow

#include "interval.h"
#include "ray.h"
