#include "SailIng/util/mesh_loader.h"
#include <stdexcept>
// #define TINYOBJLOADER_IMPLEMENTATION
#include "tiny_obj_loader.h"

namespace sail::ing {

void MeshLoader::load_obj(const std::string& file_name, MeshData& mesh) {
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
	if (!tinyobj::LoadObj(&attrib, &shapes, &materials, &warn, &err, file_name.c_str())) {
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

}// namespace sail::ing