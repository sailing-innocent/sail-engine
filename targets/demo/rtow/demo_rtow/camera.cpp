#include "camera.h"
#include "material.h"
#include "rtow.h"

namespace sail::rtow {

camera::camera(int image_width, double aspect_ratio) {
	m_image_width = image_width;
	m_image_height = static_cast<int>(image_width / aspect_ratio);
	m_image_height = m_image_height > 0 ? m_image_height : 1;
	m_aspect_ratio = aspect_ratio;
	init(image_width, aspect_ratio);
}

void camera::lookat(const point3& lookfrom, const point3& lookat, const vec3& vup) {
	m_camera_center = lookfrom;
	cz = unit_vector(lookfrom - lookat);
	cx = unit_vector(cross(vup, cz));
	cy = cross(cz, cx);
	update();
}

void camera::update() {
	init(m_image_width, m_aspect_ratio);
}

void camera::init(int image_width, double aspect_ratio) {
	auto focal_length = 1.0;
	auto tan_fov_y = std::tan(degrees_to_radians_double(vfov / 2.0));
	double viewport_height = focal_length * 2 * tan_fov_y;
	double viewport_width = aspect_ratio * viewport_height;

	auto viewport_u = viewport_width * cx;
	auto viewport_v = viewport_height * cy;
	m_pixel_dalta_u = viewport_u / (m_image_width - 1);
	m_pixel_dalta_v = viewport_v / (m_image_height - 1);
	// viewport upper left
	auto vp_ul = m_camera_center - viewport_u / 2 + viewport_v / 2 - cz * focal_length;
	m_pixel00_loc = vp_ul;
}

void camera::render(const hittable& world, std::vector<char>& image_buffer) const {
	// fill buffer
	interval colort{0.0, 1.0};
	for (auto i = 0; i < m_image_height; i++) {
		for (auto j = 0; j < m_image_width; j++) {
			color pixel_color{0.0f};
			for (auto ispp = 0; ispp < spp; ispp++) {
				auto r = get_random_ray(i, j);
				pixel_color += ray_color(r, world, max_depth);
			}
			pixel_color /= spp;

			char ir = static_cast<char>(255.999 * colort.clamp(linear_to_gamma(pixel_color[0])));
			char ig = static_cast<char>(255.999 * colort.clamp(linear_to_gamma(pixel_color[1])));
			char ib = static_cast<char>(255.999 * colort.clamp(linear_to_gamma(pixel_color[2])));
			auto index = (i * m_image_width + j) * 3;
			image_buffer[index + 0] = ir;
			image_buffer[index + 1] = ig;
			image_buffer[index + 2] = ib;
		}
	}
}

vec3 camera::pixel_sample_square() const {
	auto px = -0.5 + random_double();
	auto py = -0.5 + random_double();
	return px * m_pixel_dalta_u + py * m_pixel_dalta_v;
}

ray camera::get_random_ray(int i, int j) const {
	auto pixel_center = m_pixel00_loc + m_pixel_dalta_u * j - m_pixel_dalta_v * i;
	auto pixel_sample = pixel_center + pixel_sample_square();
	ray r{m_camera_center, pixel_sample - m_camera_center};
	return r;
}

color camera::ray_color(const ray& r, const hittable& world, int depth) const {
	if (depth <= 0) {
		return color{0.0};
	}
	hit_record rec;
	interval rayt{0.001, infinity};
	if (world.hit(r, rayt, rec)) {
		if (debug_normal) {
			return 0.5 * (rec.normal + color{1.0});
		}
		vec3 next_dir;
		if (use_matte) {
			next_dir = random_on_hemisphere(rec.normal);
			return 0.5 * ray_color(ray{rec.p, next_dir}, world, depth - 1);
		}
		if (use_lambertian) {
			next_dir = rec.normal + random_unit_vector();
			return 0.5 * ray_color(ray{rec.p, next_dir}, world, depth - 1);
		}
		if (use_material) {
			ray scattered;
			color attenuation;
			if (rec.mat->scatter(r, rec, attenuation, scattered)) {
				return attenuation * ray_color(scattered, world, depth - 1);
			}
		}
	}

	vec3 unit_direction = unit_vector(r.direction());
	auto a = 0.5 * (unit_direction[1] + 1.0);
	return (1.0 - a) * color{1.0} + a * color{0.5, 0.7, 1.0};
}

}// namespace sail::rtow