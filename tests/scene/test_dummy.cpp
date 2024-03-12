#include "core/se_test_util.h"

#include "SailDummy/one.h"

TEST_CASE("test_test")
{
  CHECK(sail::one() == 1);
}