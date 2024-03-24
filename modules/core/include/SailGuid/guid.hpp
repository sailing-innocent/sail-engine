#pragma once
/**
 * @file SailGuid/guid.hpp
 * @author sailing-innocent
 * @brief The GUID class
 * @date 2024-03-24
 */

#include "SailBase/types/guid.h"

inline SAIL_CONSTEXPR bool operator==(sail_guid_t a, sail_guid_t b) {
	bool result = true;
	result &= a.Storage0 == b.Storage0;
	result &= a.Storage1 == b.Storage1;
	result &= a.Storage2 == b.Storage2;
	result &= a.Storage3 == b.Storage3;
	return result;
}

namespace sail::guid {

struct hash {
	size_t operator()(const sail_guid_t& a) const {
		return 0;
	}
};

namespace details {

}

void make_guid();

}// namespace sail::guid