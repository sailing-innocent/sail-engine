#include "test_util.h"
#include <iostream>
#include <fstream>
#include <vector>

namespace ing::test {

int write_ppm_image() {
	const int image_width = 256;
	const int image_height = 256;
	// output buffer
	std::vector<char> image_buffer(image_width * image_height * 3, 0);
	// fill buffer
	for (auto i = 0; i < image_height; i++) {
		for (auto j = 0; j < image_width; j++) {
			auto u = (double)i / (image_height - 1);
			auto v = (double)j / (image_width - 1);
			auto r = u;
			auto g = v;
			auto b = 0.0;

			char ir = static_cast<char>(255.999 * r);
			char ig = static_cast<char>(255.999 * g);
			char ib = static_cast<char>(255.999 * b);

			auto index = (i * image_width + j) * 3;
			image_buffer[index + 0] = ir;
			image_buffer[index + 1] = ig;
			image_buffer[index + 2] = ib;
		}
	}

	// writing file
	std::ofstream ofs;
	ofs.open("D:/workspace/data/result/rtow/fig_write_image_rtow.ppm", std::ios::binary);
	ofs << "P6\n"
		<< image_width << " " << image_height << "\n255\n";
	ofs.write(image_buffer.data(), image_width * image_height * 3);
	ofs.close();
	return 0;
}

}// namespace ing::test

TEST_CASE("rtow_00") {
	REQUIRE(ing::test::write_ppm_image() == 0);
}