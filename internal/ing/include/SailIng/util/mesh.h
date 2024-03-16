#pragma once
#include <vector>
namespace sail::ing {
struct MeshData {
	std::vector<float> vertices;
	std::vector<float> normals;
	std::vector<float> texcoords;
	std::vector<int> indices;
};
}// namespace sail::ing
