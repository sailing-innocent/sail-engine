#pragma once

#include "config.h"

namespace sail::cu::reprod_gs {

#define BLOCK_SIZE (BLOCK_X * BLOCK_Y)
#define WARP_SIZE 32
#define NUM_WARPS (BLOCK_SIZE / WARP_SIZE)

// helper functions

}// namespace sail::cu::reprod_gs