/**
 * @file se_test_util.cpp
 * @brief Implementation for sail engine test utility functions
 * @author sailing-innocent
 * @date 2023-09-15
 */

#define DOCTEST_CONFIG_IMPLEMENT
#include "se_test_util.h"

#include <string>
#include <vector>

namespace se::test {

static std::vector<const char *> args;

inline void dt_remove(const char **argv_in) noexcept
{
  args.clear();
  for (; *argv_in; ++argv_in) {
    if (!std::string_view{*argv_in}.starts_with("--dt-")) {
      args.emplace_back(*argv_in);
    }
  }
  args.emplace_back(nullptr);
}

int argc() noexcept
{
  return static_cast<int>(args.size());
}
const char *const *argv() noexcept
{
  return args.data();
}

}  // namespace se::test

int main(int argc, const char **argv)
{
  doctest::Context context(argc, argv);
  se::test::dt_remove(argv);
  auto test_result = context.run();
  if (context.shouldExit()) {
    return test_result;
  }
  return test_result;
}