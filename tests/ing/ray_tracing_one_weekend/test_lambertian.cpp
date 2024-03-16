#include "test_util.h"
#include "rtow.h"
#include "hittable.h"
#include "hittable_list.h"
#include "sphere.h"
#include "camera.h"

#include <string>
#include <vector>

namespace ing::test06 {

using namespace ing::rtow;

int test_lambertian() {
	double aspect_ratio = 16.0 / 9.0;
	int image_width = 512;
	int image_height = static_cast<int>(image_width / aspect_ratio);
	image_height = image_height > 0 ? image_height : 1;
	camera cam{image_width, aspect_ratio};
	cam.use_lambertian = true;
	// world
	hittable_list world;
	world.add(std::make_shared<sphere>(point3{0.0, 0.0, -1.0}, 0.5));
	world.add(std::make_shared<sphere>(point3{0.0, -100.5, -1.0}, 100.0));
	// output buffer
	std::vector<char> image_buffer(image_width * image_height * 3, 0);

	std::vector<int> exp_depths = {10, 50};
	std::vector<int> exp_spps = {10, 100};

	for (auto depth : exp_depths) {
		for (auto spp : exp_spps) {
			cam.max_depth = depth;
			cam.spp = spp;
			cam.render(world, image_buffer);
			// writing file
			std::string fname = "D:/workspace/data/result/rtow/fig_lambertian_rtow_" + std::to_string(depth) + "_" + std::to_string(spp) + ".ppm";
			write_image(fname, image_buffer, image_width, image_height);
		}
	}

	return 0;
}

}// namespace ing::test06

TEST_CASE("rtow_08") {
	REQUIRE(ing::test06::test_lambertian() == 0);
}