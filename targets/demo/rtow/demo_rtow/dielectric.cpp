#include "material.h"
#include <iostream>

namespace sail::rtow {

bool dielectric::scatter(const ray& r_in, const hit_record& rec, color& attenuation, ray& scattered) const {
	attenuation = color{1.0, 1.0, 1.0};
	double refraction_ratio = rec.front_face ? (1.0 / ref_idx) : ref_idx;
	vec3 unit_direction = unit_vector(r_in.direction());
	double cos_theta = fmin(dot(-unit_direction, rec.normal), 1.0);
	double sin_theta = std::sqrt(1.0 - cos_theta * cos_theta);
	bool cannot_refract = refraction_ratio * sin_theta > 1.0;
	vec3 direction;
	if (cannot_refract || reflectance(cos_theta, refraction_ratio) > random_double()) {
		direction = reflect(unit_direction, rec.normal);
	} else {
		direction = refract(unit_direction, rec.normal, refraction_ratio);
	}
	scattered = ray{rec.p, direction};
	return true;
}

}// namespace sail::rtow