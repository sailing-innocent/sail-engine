#pragma once
/**
 * @file dstorage_interface.h
 * @brief The DirectStorage Interface 
 * @author sailing-innocent
 * @date 2024-05-15
 */
#include "SailInno/config.h"
#include <luisa/luisa-compute.h>

namespace sail::inno {

enum class DStorageSrcType : uint64_t {
	File = 0,
	Memory = 1
};
class SAIL_INNO_API IOFile {
public:
	class Handle {
		friend class IOFile;
	};
};

class SAIL_INNO_API DStorageStream {
public:
	void* queue{nullptr};
	DStorageSrcType src_type;
	static void init(
		luisa::filesystem::path const& runtime_dir);
};

}// namespace sail::inno