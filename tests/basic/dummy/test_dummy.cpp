#include "SailDummy/one.h"
#include "se_test_util.h"

TEST_CASE("dummy")
{
  CHECK(sail::one() == 1);
}