#pragma once

/**
 * @file service.h
 * @brief The IOService 
 * @author sailing-innocent
 * @date 2024-05-15
 */
#include "SailInno/config.h"

namespace sail::inno {

class SAIL_INNO_API IOService {
	friend class ioservice_detail::CallbackThread;
	uint64_t _self_idx;
};

}// namespace sail::inno