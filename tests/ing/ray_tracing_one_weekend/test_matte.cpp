#include "test_util.h"
#include "rtow.h"
#include "hittable.h"
#include "hittable_list.h"
#include "sphere.h"
#include "camera.h"

namespace ing::test05 {

using namespace ing::rtow;

int test_matte() {
	double aspect_ratio = 16.0 / 9.0;
	int image_width = 512;
	int image_height = static_cast<int>(image_width / aspect_ratio);
	image_height = image_height > 0 ? image_height : 1;
	camera cam{image_width, aspect_ratio};
	cam.use_matte = true;
	// world
	hittable_list world;
	world.add(std::make_shared<sphere>(point3{0.0, 0.0, -1.0}, 0.5));
	world.add(std::make_shared<sphere>(point3{0.0, -100.5, -1.0}, 100.0));
	// output buffer
	std::vector<char> image_buffer(image_width * image_height * 3, 0);

	cam.max_depth = 1;
	cam.render(world, image_buffer);
	// writing file
	std::string fname = "D:/workspace/data/result/rtow/fig_matte_rtow_1.ppm";
	write_image(fname, image_buffer, image_width, image_height);

	cam.max_depth = 10;
	fname = "D:/workspace/data/result/rtow/fig_matte_rtow_10.ppm";
	cam.render(world, image_buffer);
	write_image(fname, image_buffer, image_width, image_height);

	cam.max_depth = 50;
	fname = "D:/workspace/data/result/rtow/fig_matte_rtow_50.ppm";
	cam.render(world, image_buffer);
	write_image(fname, image_buffer, image_width, image_height);

	return 0;
}

}// namespace ing::test05

TEST_CASE("rtow_07") {
	REQUIRE(ing::test05::test_matte() == 0);
}