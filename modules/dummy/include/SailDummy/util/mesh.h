#pragma once
#include <vector>
namespace sail::dummy {
using std::vector;
struct MeshData {
	vector<float> vertices;
	vector<float> normals;
	vector<float> texcoords;
	vector<int> indices;

	MeshData() = default;
	// delete copy
	MeshData(const MeshData&) = delete;
	MeshData& operator=(const MeshData&) = delete;
	// support move
	MeshData(MeshData&&) = default;
	MeshData& operator=(MeshData&&) = default;
};

}// namespace sail::dummy
