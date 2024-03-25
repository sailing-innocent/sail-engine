#pragma once

#include "rtow.h"
#include "material.h"

namespace sail::rtow {

class material;

class hit_record {
public:
	point3 p;
	vec3 normal;
	std::shared_ptr<material> mat;
	double t;
	bool front_face;

	void set_front_face(const ray& r, const vec3& outward_normal) {
		front_face = dot(r.direction(), outward_normal) < 0;
		normal = front_face ? outward_normal : -outward_normal;
	}
};

class hittable {
public:
	virtual ~hittable() = default;
	virtual bool hit(const ray& r, interval rayt, hit_record& rec) const = 0;
};

}// namespace sail::rtow