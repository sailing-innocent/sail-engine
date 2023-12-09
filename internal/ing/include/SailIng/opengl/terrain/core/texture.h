#pragma once

/** 
 * @file terrain/texture.h
 * @author sailing-innocent
 * @date 2023-08-30
 * @brief the terrain texture
 */

#include <vector>
#include <string>

namespace ing::terrain
{

unsigned int texture_from_file(const char* path, const std::string& directory, bool gamma = false);
unsigned int load_cube_map(std::vector<std::string> faces);
unsigned int generate_texture_2d(int w, int h);
unsigned int generate_texture_3d(int w, int h, int d);
void         bind_texture_2d(unsigned int tex, int unit = 0);

}  // namespace ing::terrain