#include "SailInno/util/misc/bit_helper.h"

namespace sail::inno::util {

uint32_t get_higher_msb(uint32_t n)
{
  uint32_t msb = sizeof(n) * 4;
  uint32_t step = msb;
  while (step > 1) {
    step /= 2;
    if (n >> msb)
      msb += step;
    else
      msb -= step;
  }
  if (n >> msb)
    msb++;
  return msb;
}

}  // namespace sail::inno::util