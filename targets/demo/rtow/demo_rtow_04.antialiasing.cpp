#include "demo_rtow/hittable_list.h"
#include "demo_rtow/sphere.h"
#include "demo_rtow/camera.h"

#include <fstream>
#include <string>
#include <filesystem>
#include <iostream>

using namespace sail::rtow;

int main(int argc, char** argv) {
	std::string odir = argv[1];
	std::string oname = argv[2];
	std::filesystem::path odir_path(odir);
	std::filesystem::create_directories(odir_path);
	std::filesystem::path of_path(odir + "/" + oname + ".ppm");

	double aspect_ratio = 16.0 / 9.0;
	int image_width = 512;
	int image_height = static_cast<int>(image_width / aspect_ratio);
	image_height = image_height > 0 ? image_height : 1;
	camera cam{image_width, aspect_ratio};
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
	ofs.open(of_path, std::ios::binary);
	std::cout << "Writing image to " << of_path << std::endl;
	ofs << "P6\n"
		<< image_width << " " << image_height << "\n255\n";
	ofs.write(image_buffer.data(), image_width * image_height * 3);
	ofs.close();
	return 0;
}
