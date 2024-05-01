#pragma once
/**
 * @file loader.h
 * @brief The Dummy Loader helpers
 * @author sailing-innocent
 * @date 2024-05-01
 */
#include "SailDummy/config.h"

#include <vector>
#include <string_view>

namespace sail::dummy {

using std::string;
using std::string_view;
using std::vector;
class MeshData;
void SAIL_DUMMY_API read_file(string_view fname, vector<char>& buffer);
void SAIL_DUMMY_API load_mesh_from_obj(string_view fname, MeshData& mesh);

}// namespace sail::dummy