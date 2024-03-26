#pragma once

#include "rtow.h"
#include "hittable.h"
#include <vector>

namespace sail::rtow {

class camera {
public:
	int spp = 10;
	int max_depth = 10;
	double vfov = 90.0;
	bool debug_normal = false;
	bool use_lambertian = false;
	bool use_matte = false;
	bool use_material = false;

	camera(int image_width, double aspect_ratio);
	void lookat(const point3& lookfrom, const point3& lookat, const vec3& vup);

	void render(const hittable& world, std::vector<char>& image_buffer) const;
	void init(int image_width, double aspect_ratio);
	void update();

private:
	int m_image_width = 200;
	int m_image_height = 100;
	double m_aspect_ratio = 16.0 / 9.0;
	point3 m_camera_center = point3{0.0};
	vec3 cx = vec3{1.0, 0.0, 0.0};
	vec3 cy = vec3{0.0, 1.0, 0.0};
	vec3 cz = vec3{0.0, 0.0, 1.0};

	vec3 m_pixel_dalta_u, m_pixel_dalta_v;
	point3 m_pixel00_loc;

	color ray_color(const ray& r, const hittable& world, int depth = 1) const;
	ray get_random_ray(int i, int j) const;
	vec3 pixel_sample_square() const;
};

}// namespace sail::rtow