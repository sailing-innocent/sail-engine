#pragma once
#include "SailInno/core/runtime.h"
#include <luisa/vstl/common.h>

namespace sail::inno {

class SAIL_INNO_API Pipeline : public LuisaModule {
public:
	mutable vstd::LMDB db;
};

}// namespace sail::inno