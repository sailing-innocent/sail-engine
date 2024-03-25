#include "demo_rtow/rtow.h"
#include "demo_rtow/hittable.h"
#include "demo_rtow/hittable_list.h"
#include "demo_rtow/sphere.h"
#include "demo_rtow/camera.h"
#include "demo_rtow/material.h"

using namespace sail::rtow;

int main(int argc, char** argv) {

	std::string odir = argv[1];
	std::string oname = argv[2];
	std::filesystem::path odir_path(odir);
	std::filesystem::create_directories(odir_path);
	std::filesystem::path of_path(odir + "/" + oname + ".ppm");

	double aspect_ratio = 16.0 / 9.0;
	int image_width = 1200;
	int image_height = static_cast<int>(image_width / aspect_ratio);
	image_height = image_height > 0 ? image_height : 1;
	camera cam{image_width, aspect_ratio};
	cam.vfov = 20.0;
	cam.lookat(
		point3{13, -3, 2},
		point3{0, 0, 0},
		vec3{0, 0, 1});
	cam.use_material = true;
	cam.max_depth = 50;
	cam.spp = 100;

	hittable_list world;
	auto material_ground = std::make_shared<lambertian>(color{0.5, 0.5, 0.5});
	world.add(std::make_shared<sphere>(point3{0.0, 0.0, -1000.0}, 1000.0, material_ground));

	for (int a = -11; a < 11; a++) {
		for (int b = -11; b < 11; b++) {
			auto choose_mat = random_double();
			point3 center{a + 0.9 * random_double(), b + 0.9 * random_double(), 0.2};
			auto dist = center - point3{4, 0, 0.2};
			if (dot(dist, dist) > 0.81) {
				std::shared_ptr<material> sphere_material;
				if (choose_mat < 0.8) {
					// diffuse
					auto albedo = random_vec3() * random_vec3();
					sphere_material = std::make_shared<lambertian>(albedo);
					world.add(std::make_shared<sphere>(center, 0.2, sphere_material));
				} else if (choose_mat < 0.95) {
					// metal
					auto albedo = random_vec3(0.5, 1);
					auto fuzz = random_double(0, 0.5);
					sphere_material = std::make_shared<metal>(albedo, fuzz);
					world.add(std::make_shared<sphere>(center, 0.2, sphere_material));
				} else {
					// glass
					sphere_material = std::make_shared<dielectric>(1.5);
					world.add(std::make_shared<sphere>(center, 0.2, sphere_material));
				}
			}
		}
	}
	auto material1 = std::make_shared<dielectric>(1.5);
	world.add(std::make_shared<sphere>(point3{0, 0, 1}, 1.0, material1));

	auto material2 = std::make_shared<lambertian>(color{0.4, 0.2, 0.1});
	world.add(std::make_shared<sphere>(point3{-4, 0, 1}, 1.0, material2));

	auto material3 = std::make_shared<metal>(color{0.7, 0.6, 0.5}, 0.0);
	world.add(std::make_shared<sphere>(point3{4, 0, 1}, 1.0, material3));

	std::vector<char> image_buffer(image_width * image_height * 3, 0);
	cam.render(world, image_buffer);

	write_image(of_path, image_buffer, image_width, image_height);
	return 0;
}
