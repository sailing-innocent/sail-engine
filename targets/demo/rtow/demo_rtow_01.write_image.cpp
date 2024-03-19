#include <fstream>
#include <vector>
#include <string>
#include <filesystem>

int main(int argc, char** argv) {
	const int image_width = 256;
	const int image_height = 256;

	std::string odir = argv[1];
	std::string oname = argv[2];
	std::filesystem::path odir_path(odir);
	std::filesystem::create_directories(odir_path);
	std::filesystem::path of_path(odir + "/" + oname + ".ppm");

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
	ofs.open(of_path, std::ios::binary);
	ofs << "P6\n"
		<< image_width << " " << image_height << "\n255\n";
	ofs.write(image_buffer.data(), image_width * image_height * 3);
	ofs.close();
	return 0;
}
