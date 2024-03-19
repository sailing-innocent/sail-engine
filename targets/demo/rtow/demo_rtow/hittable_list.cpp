#include "hittable_list.h"

namespace sail::rtow {

bool hittable_list::hit(const ray& r, interval rayt, hit_record& rec) const {
	hit_record temp_rec;
	bool hit_anything = false;
	auto closest_so_far = rayt.m_max;
	for (const auto& object : objects) {
		interval newt{rayt.m_min, closest_so_far};
		if (object->hit(r, newt, temp_rec)) {
			hit_anything = true;
			closest_so_far = temp_rec.t;
			rec = temp_rec;
		}
	}
	return hit_anything;
}

}// namespace sail::rtow