#include "SailIng/opengl/terrain/core/texture.h"

namespace ing::terrain {

unsigned int texture_from_file(const char* path, const std::string& directory, bool gamma) {
	std::string filename{path};
	filename = directory + '/' + filename;

	unsigned int texture_id;

	return texture_id;
}

unsigned int load_cube_map(std::vector<std::string> faces) {
	unsigned int texture_id;

	return texture_id;
}

unsigned int generate_texture_2d(int w, int h) {
	unsigned int tex_output;

	return tex_output;
}

unsigned int generate_texture_3d(int w, int h, int d) {
	unsigned int tex_output;

	return tex_output;
}

void bind_texture_2d(unsigned int tex, int unit) {}

}// namespace ing::terrain
