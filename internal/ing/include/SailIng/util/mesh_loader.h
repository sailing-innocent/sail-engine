#pragma once

#include <string>

namespace sail::ing {

class MeshLoader {
public:
	void load_obj(const std::string& file_name, MeshData& mesh);
};// class MeshLoader

}// namespace sail::ing