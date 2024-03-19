#pragma once
#include "SailBase/config.h"

#ifdef __cplusplus
#include <initializer_list>
#endif

using sail_guid_t = struct sail_guid_t {
#ifdef __cplusplus

#endif
	uint32_t Storage0 SAIL_IF_CPP(= 0);
	uint32_t Storage1 SAIL_IF_CPP(= 0);
	uint32_t Storage2 SAIL_IF_CPP(= 0);
	uint32_t Storage3 SAIL_IF_CPP(= 0);
};

SAIL_EXTERN_C void skr_make_guid(sail_guid_t* out_guid);
