#pragma once
#include "rtm/vector4f.h"
#include "SailBase/types.h"

namespace sail::math {

inline rtm::vector4f load(const sail_float4_t v) {
	return rtm::vector4f{v.x, v.y, v.z, v.w};
}
inline void store(rtm::vector4f v, sail_float4_t& result) {
	rtm::vector_store(v, &result.x);
}

}// namespace sail::math