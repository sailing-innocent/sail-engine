#pragma once

#include "vec3.h"

namespace sail::rtow {

class ray {

public:
	ray() = default;
	explicit ray(const point3& origin, const vec3& direction) : m_orig(origin), m_dir(direction) {}

	point3 at(double t) const {
		return m_orig + t * m_dir;
	}
	point3 origin() const {
		return m_orig;
	}
	vec3 direction() const {
		return m_dir;
	}

private:
	point3 m_orig;
	vec3 m_dir;
};

}// namespace sail::rtow