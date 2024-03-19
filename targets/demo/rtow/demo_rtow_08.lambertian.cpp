#include "demo_rtow/rtow.h"
#include "demo_rtow/hittable_list.h"
#include "demo_rtow/sphere.h"
#include "demo_rtow/camera.h"

#include <string>
#include <vector>
#include <filesystem>

using namespace sail::rtow;

int main(int argc, char** argv) {
	std::string odir = argv[1];
	std::string oname = argv[2];
	std::filesystem::path odir_path(odir);
	std::filesystem::create_directories(odir_path);

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
			std::filesystem::path of_path(odir + "/" + oname + "_" + std::to_string(depth) + "_" + std::to_string(spp) + ".ppm");
			write_image(of_path, image_buffer, image_width, image_height);
		}
	}
	return 0;
}
