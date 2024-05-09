#pragma once

#include "rasterizer.h"
#include <cuda_runtime_api.h>
#include <memory>

namespace sail::cu::reprod_gs {

struct GeometryState {
	void from_chunk(char*& chunk, size_t P) {
	}
};

struct ImageState {
};

struct TileState {
};

}// namespace sail::cu::reprod_gs