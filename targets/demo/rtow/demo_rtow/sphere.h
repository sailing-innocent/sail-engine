#pragma once
#include "hittable.h"

namespace sail::rtow {

class sphere : public hittable {
public:
	sphere() {}
	sphere(point3 center, double radius, std::shared_ptr<material> mat = nullptr) : m_center(center), m_radius(radius), m_mat(mat) {}

	virtual bool hit(const ray& r, interval rayt, hit_record& rec) const override;

private:
	point3 m_center;
	double m_radius;
	std::shared_ptr<material> m_mat;
};

}// namespace sail::rtow