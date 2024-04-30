#pragma once
/**
 * @file stl.h
 * @brief Wrapper for STL containers
 * @author sailing-innocent
 * @date 2024-04-30
 */
#include <vector>
#include <memory>
#include <string>

namespace sail {
template<typename T>
using vector = std::vector<T>;
template<typename T>
using unique_ptr = std::unique_ptr<T>;
template<typename T>
unique_ptr<T> make_unique(T&& inst) {
	return std::make_unique<T>(std::forward<T>(inst));
}

template<typename T>
using shared_ptr = std::shared_ptr<T>;
template<typename T>
shared_ptr<T> make_shared(T&& inst) {
	return std::make_shared<T>(std::forward<T>(inst));
}

using string = std::string;

}// namespace sail