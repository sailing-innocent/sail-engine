#include "test_util.h"
#include "rtow.h"
#include "hittable.h"
#include "hittable_list.h"
#include "sphere.h"

#include <fstream>

namespace ing::test02 {

using namespace ing::rtow;

color ray_color(const ray& r, const hittable& world) {
	hit_record rec;
	interval rayt{0.0, infinity};
	if (world.hit(r, rayt, rec)) {
		return 0.5 * (rec.normal + color{1.0});
	}

	vec3 unit_direction = unit_vector(r.direction());
	auto a = 0.5 * (unit_direction[1] + 1.0);
	return (1.0 - a) * color{1.0} + a * color{0.5, 0.7, 1.0};
}

int test_sphere_normal() {
	double aspect_ratio = 16.0 / 9.0;
	int image_width = 512;
	int image_height = static_cast<int>(image_width / aspect_ratio);
	image_height = image_height > 0 ? image_height : 1;
	// viewport
	double viewport_height = 2.0;
	double viewport_width = aspect_ratio * viewport_height;
	auto camera_center = point3{0.0};
	auto viewport_u = vec3{viewport_width, 0.0, 0.0};
	auto viewport_v = vec3{0.0, viewport_height, 0.0};
	auto pixel_dalta_u = viewport_u / (image_width - 1);
	auto pixel_dalta_v = viewport_v / (image_height - 1);
	auto focal_length = 1.0;
	// viewport upper left
	auto vp_ul = camera_center - viewport_u / 2 + viewport_v / 2 - vec3{0.0, 0.0, focal_length};
	auto pixel00_loc = vp_ul;

	// world
	hittable_list world;
	world.add(std::make_shared<sphere>(point3{0.0, 0.0, -1.0}, 0.5));

	// output buffer
	std::vector<char> image_buffer(image_width * image_height * 3, 0);
	// fill buffer

	for (auto i = 0; i < image_height; i++) {
		for (auto j = 0; j < image_width; j++) {
			ray r{camera_center, pixel00_loc + pixel_dalta_u * j - pixel_dalta_v * i - camera_center};
			color pixel_color = ray_color(r, world);
			char ir = static_cast<char>(255.999 * pixel_color[0]);
			char ig = static_cast<char>(255.999 * pixel_color[1]);
			char ib = static_cast<char>(255.999 * pixel_color[2]);
			auto index = (i * image_width + j) * 3;
			image_buffer[index + 0] = ir;
			image_buffer[index + 1] = ig;
			image_buffer[index + 2] = ib;
		}
	}

	// writing file
	std::ofstream ofs;
	ofs.open("D:/workspace/data/result/rtow/fig_sphere_normal_rtow.ppm", std::ios::binary);
	ofs << "P6\n"
		<< image_width << " " << image_height << "\n255\n";
	ofs.write(image_buffer.data(), image_width * image_height * 3);
	ofs.close();
	return 0;
}

}// namespace ing::test02

TEST_CASE("rtow_04") {
	REQUIRE(ing::test02::test_sphere_normal() == 0);
}