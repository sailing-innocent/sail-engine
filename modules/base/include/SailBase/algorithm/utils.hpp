#pragma once
#include "SailBase/config.h"
#include <concept>
#include <type_traits>

// TODO: functors

namespace sail {
// map functor

// compare functor
#define SAIL_DEF_COMPARE_FUNCTOR(__NAME, __OP)                                \
	template<typename T = void>                                               \
	struct __Name {                                                           \
		SAIL_INLINE constexpr bool operator()(const T& a, const T& b) const { \
			return a __OP b;                                                  \
		}                                                                     \
	};                                                                        \
	template<>                                                                \
	struct __Name<void> {                                                     \
		template<typename T, typename U>                                      \
		SAIL_INLINE constexpr bool operator()(const T& a, const U& b) const { \
			return a __OP b;                                                  \
		}                                                                     \
	};

SAIL_DEF_COMPARE_FUNCTOR(Less, <)

#undef SAIL_DEF_COMPARE_FUNCTOR

}// namespace sail