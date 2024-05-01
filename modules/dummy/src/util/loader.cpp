/**
 * @file loader.cpp
 * @brief The implementation of Dummy Loader
 * @author sailing-innocent
 * @date 2024-05-01
 */

#include "SailDummy/util/loader.h"
#include "SailDummy/util/mesh.h"
#include <fstream>
#include <iostream>
#include "tiny_obj_loader.h"

namespace sail::dummy {

using std::ifstream;
using std::ios;
using std::string;
using std::string_view;
using std::vector;

void read_file(string_view fname, vector<char>& buffer) {
	ifstream file(fname.data(), ios::ate | ios::binary);
	if (!file.is_open()) {
		throw std::runtime_error("failed to open file: " + string(fname));
	}
	size_t fileSize = (size_t)file.tellg();
	buffer.resize(fileSize);
	file.seekg(0);
	file.read(buffer.data(), fileSize);
	file.close();
}

void load_mesh_from_obj(string_view fname, MeshData& mesh) {
	mesh.vertices = {
		-0.5f, -0.5f, 0.0f,// point A
		0.0f, -0.5f, 0.0f, // point B
		0.0f, 0.5f, 0.0f,  // point C
		0.5f, -0.5f, 0.0f, // point D
	};
	mesh.indices = {
		0, 1, 2, 1, 3, 2};
	tinyobj::attrib_t attrib;
	std::vector<tinyobj::shape_t> shapes;
	std::vector<tinyobj::material_t> materials;// TODO: use materials
	std::string warn, err;
	if (!tinyobj::LoadObj(&attrib, &shapes, &materials, &warn, &err, fname.data())) {
		throw std::runtime_error(warn + err);
	}

	size_t vertex_count = attrib.GetVertices().size();
	mesh.vertices.resize(vertex_count);

	for (size_t i = 0; i < vertex_count; i++) {
		mesh.vertices[i] = attrib.GetVertices()[i];
	}
	for (auto i = 0; i < shapes.size(); i++) {
		size_t index_count = shapes[i].mesh.indices.size();
		for (auto j = 0; j < index_count; j++) {
			mesh.indices.push_back(shapes[i].mesh.indices[j].vertex_index);
		}
	}
}

}// namespace sail::dummy