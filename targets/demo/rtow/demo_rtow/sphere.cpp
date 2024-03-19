#include "sphere.h"

namespace sail::rtow {

bool sphere::hit(const ray& r, interval rayt, hit_record& rec) const {
	vec3 oc = r.origin() - m_center;
	auto a = dot(r.direction(), r.direction());
	auto half_b = dot(oc, r.direction());
	auto c = dot(oc, oc) - m_radius * m_radius;
	auto discriminant = half_b * half_b - a * c;

	if (discriminant > 0) {
		auto root = sqrt(discriminant);
		auto temp = (-half_b - root) / a;
		if (rayt.surrounds(temp)) {
			rec.t = temp;
			rec.p = r.at(rec.t);
			rec.mat = m_mat;
			vec3 outward_normal = (rec.p - m_center) / m_radius;
			rec.set_front_face(r, outward_normal);
			return true;
		}
		temp = (-half_b + root) / a;
		if (rayt.surrounds(temp)) {
			rec.t = temp;
			rec.p = r.at(rec.t);
			rec.mat = m_mat;
			vec3 outward_normal = (rec.p - m_center) / m_radius;
			rec.set_front_face(r, outward_normal);
			return true;
		}
	}
	return false;
}

}// namespace sail::rtow