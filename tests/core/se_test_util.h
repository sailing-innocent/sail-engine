/**
 * @file se_test_util.h
 * @brief Header for sail engine test utility functions and macros
 * @author sailing-innocent
 * @date 2023-09-15
 */

#pragma once

#include <doctest.h>

namespace se::test {
[[nodiscard]] int argc() noexcept;
[[nodiscard]] const char *const *argv() noexcept;
}  // namespace se::test
