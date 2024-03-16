// commonly used utility functions
#include "SailIng/util/file_loader.h"
#include "stb_image.h"
#include "stb_image_write.h"

namespace sail::ing {

std::vector<char> readfile(const std::string& filename) {
	// std::cout << "Is Opening File: " + filename << std::endl;
	std::ifstream file(filename, std::ios::ate | std::ios::binary);
	if (!file.is_open()) {
		throw std::runtime_error("failed to open file: " + filename);
	}
	size_t fileSize = (size_t)file.tellg();
	std::vector<char> buffer(fileSize);
	file.seekg(0);
	file.read(buffer.data(), fileSize);
	file.close();
	return buffer;
}

}// namespace sail::ing
