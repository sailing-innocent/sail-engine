#include "test_util.h"
#include "rtow.h"
#include "hittable.h"
#include "hittable_list.h"
#include "sphere.h"
#include "camera.h"
#include "material.h"

#include <string>
#include <vector>

namespace ing::test {

using namespace ing::rtow;

int test_dielectric() {
	double aspect_ratio = 16.0 / 9.0;
	int image_width = 512;
	int image_height = static_cast<int>(image_width / aspect_ratio);
	image_height = image_height > 0 ? image_height : 1;
	camera cam{image_width, aspect_ratio};
	cam.use_material = true;
	// world
	hittable_list world;
	auto material_ground = std::make_shared<lambertian>(color{0.8, 0.8, 0.0});
	auto material_center = std::make_shared<lambertian>(color{0.1, 0.2, 0.5});
	auto material_left = std::make_shared<dielectric>(1.5);
	auto material_right = std::make_shared<metal>(color{0.8, 0.6, 0.2}, 0.0);

	world.add(std::make_shared<sphere>(point3{0.0, -100.5, -1.0}, 100.0, material_ground));
	world.add(std::make_shared<sphere>(point3{0.0, 0.0, -1.0}, 0.5, material_center));
	world.add(std::make_shared<sphere>(point3{-1.0, 0.0, -1.0}, 0.5, material_left));
	// hollow glass sphere
	world.add(std::make_shared<sphere>(point3{-1.0, 0.0, -1.0}, -0.4, material_left));
	world.add(std::make_shared<sphere>(point3{1.0, 0.0, -1.0}, 0.5, material_right));

	// output buffer
	std::vector<char> image_buffer(image_width * image_height * 3, 0);

	std::vector<int> exp_depths = {50};
	std::vector<int> exp_spps = {100};

	for (auto depth : exp_depths) {
		for (auto spp : exp_spps) {
			cam.max_depth = depth;
			cam.spp = spp;
			cam.render(world, image_buffer);
			// writing file
			std::string fname = "D:/workspace/data/result/rtow/fig_dielectric_rtow_" + std::to_string(depth) + "_" + std::to_string(spp) + ".ppm";
			write_image(fname, image_buffer, image_width, image_height);
		}
	}

	return 0;
}

}// namespace ing::test

TEST_CASE("rtow_10") {
	REQUIRE(ing::test::test_dielectric() == 0);
}