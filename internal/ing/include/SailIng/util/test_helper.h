#pragma once
/**
 * @file util/test_helper.h
 * @brief The Test Helper Util for ING
 * @date 2024-03-13
 * @author sailing-innocent
 */

#include <glm/glm.hpp>  // vec2
#include <iostream>

namespace sail::ing {

template<typename T, int I> void print_glm_vec(const glm::vec<I, T> &vec)
{
  for (auto i = 0; i < I; i++) {
    std::cout << vec[i] << " ";
  }
  std::cout << std::endl;
}

}  // namespace sail::ing