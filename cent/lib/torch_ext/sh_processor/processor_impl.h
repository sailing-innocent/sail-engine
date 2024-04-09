#pragma once

#include "processor.h"

namespace CudaSHProcessor {
template<typename T>
static void obtain(char*& chunk, T*& ptr, std::size_t count,
				   std::size_t alignment) {
	std::size_t offset =
		(reinterpret_cast<std::uintptr_t>(chunk) + alignment - 1) &
		~(alignment - 1);
	ptr = reinterpret_cast<T*>(offset);
	chunk = reinterpret_cast<char*>(ptr + count);
}
struct GeometryState {
	bool* clamped;
	static GeometryState fromChunk(char*& chunk, size_t P);
};
};// namespace CudaSHProcessor