#pragma once
#include "SailInno/core/runtime.h"
namespace sail::inno {
using namespace luisa;
using namespace luisa::compute;
namespace io_service_detail {
struct CallbackThread;
}// namespace io_service_detail

class SAIL_INNO_API IOService : public LuisaModule {
};
}// namespace sail::inno