#pragma once
#include "hittable.h"
#include <memory>
#include <vector>

namespace sail::rtow {

class hittable_list : public hittable {
public:
	std::vector<std::shared_ptr<hittable>> objects;

	hittable_list() {}
	hittable_list(std::shared_ptr<hittable> object) {
		add(object);
	}
	void clear() {
		objects.clear();
	}

	void add(std::shared_ptr<hittable> object) {
		objects.push_back(object);
	}
	bool hit(const ray& r, interval rayt, hit_record& rec) const override;
};

}// namespace sail::rtow