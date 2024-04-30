#pragma once
/**
 * @file stl.h
 * @brief Wrapper for STL containers
 * @author sailing-innocent
 * @date 2024-04-30
 */
#include <vector>
#include <memory>
namespace sail {
template<typename T>
using vector = std::vector<T>;
template<typename T>
using unique_ptr = std::unique_ptr<T>;
template<typename T>
using shared_ptr = std::shared_ptr<T>;
}// namespace sail