#pragma once

namespace sail::cu {

template<typename T>
static void obtain(char*& chunk, T*& ptr, size_t count, size_t alignment) {
	size_t offset = (reinterpret_cast<uintptr_t>(chunk) + alignment - 1) & ~(alignment - 1);
	ptr = reinterpret_cast<T*>(offset);
	chunk = reinterpret_cast<char*>(ptr + count);
}

template<typename T>
size_t required(size_t P) {
	char* chunk_size = nullptr;
	T::from_chunk(chunk_size, P);
	return ((size_t)chunk_size) + 128;
}

}// namespace sail::cu