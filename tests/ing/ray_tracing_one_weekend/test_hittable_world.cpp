#include "test_util.h"
#include "rtow.h"
#include "hittable.h"
#include "hittable_list.h"
#include "sphere.h"
#include "camera.h"

#include <fstream>

namespace ing::test03 {

using namespace ing::rtow;

int test_hittable_world() {
	double aspect_ratio = 16.0 / 9.0;
	int image_width = 512;
	int image_height = static_cast<int>(image_width / aspect_ratio);
	image_height = image_height > 0 ? image_height : 1;
	camera cam{image_width, aspect_ratio};
	cam.spp = 1;
	cam.debug_normal = true;
	// world
	hittable_list world;
	world.add(std::make_shared<sphere>(point3{0.0, 0.0, -1.0}, 0.5));
	world.add(std::make_shared<sphere>(point3{0.0, -100.5, -1.0}, 100.0));
	// output buffer
	std::vector<char> image_buffer(image_width * image_height * 3, 0);

	cam.render(world, image_buffer);
	// writing file
	std::ofstream ofs;
	ofs.open("D:/workspace/data/result/rtow/fig_hittable_world_rtow.ppm", std::ios::binary);
	ofs << "P6\n"
		<< image_width << " " << image_height << "\n255\n";
	ofs.write(image_buffer.data(), image_width * image_height * 3);
	ofs.close();
	return 0;
}

}// namespace ing::test03

TEST_CASE("rtow_05") {
	REQUIRE(ing::test03::test_hittable_world() == 0);
}