/**
 * @file se_test_util.h
 * @brief Header for sail engine test utility functions and macros
 * @author sailing-innocent
 * @date 2023-09-15
 */

#pragma once

#include <doctest.h>
#include <span>

namespace sail::test {
[[nodiscard]] int argc() noexcept;
[[nodiscard]] const char* const* argv() noexcept;

[[nodiscard]] bool float_span_equal(std::span<float> a, std::span<float> b);

}// namespace sail::test
