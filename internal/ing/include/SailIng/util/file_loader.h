#pragma once
/**
 * @file: util/file_loader
 * @author: sailing-innocent
 * @create: 2022-11-06
 * @desp: The Common Utility Functions
*/

#include <vector>
#include <string>
#include <fstream>
#include <sstream>
#include <iostream>

namespace sail::ing {

std::vector<char> readfile(const std::string& filename);

}// namespace sail::ing
