#include "SailDummy/one.h"
#include "test_util.h"

TEST_CASE("dummy")
{
  CHECK(sail::one() == 1);
}