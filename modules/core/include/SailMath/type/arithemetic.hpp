#pragma once
/**
 * @file arithemetic.hpp
 * @brief The Sail Supported Arithemetic Type
 * @author sailing-innocent
 * @date 2024-04-29
 */

#include <concepts>
#include <type_traits>

namespace sail {

// TODO: complex number
// TODO: quaternion
// TODO：dual number

template<typename T>
static constexpr bool is_arithemetic_v = std::is_integral_v<T> || std::is_floating_point_v<T>;

template<typename T>
concept ArithemeticT = is_arithemetic_v<T>;

}// namespace sail