#pragma once
#include "SailBase/config.h"
#include "SailBase/config/key_words.h"

#ifdef __cplusplus
#define SAIL_DECLARE_VEC2_BODY(TT, NAME) \
	TT x, y;                             \
	SAIL_FORCEINLINE bool operator==(const NAME& vec) const { return x == vec.x && y == vec.y; }
#else
#define SAIL_DECLARE_VEC2_BODY(TT, NAME) TT x, y;
#endif

using sail_float2_t = struct sail_float2_t {
	SAIL_DECLARE_VEC2_BODY(float, sail_float2_t)
};

using sail_float4_t = struct SAIL_ALIGNAS(16) sail_float4_t {
	struct {
		float x SAIL_IF_CPP(= 0.f);
		float y SAIL_IF_CPP(= 0.f);
		float z SAIL_IF_CPP(= 0.f);
		float w SAIL_IF_CPP(= 0.f);
	};
};