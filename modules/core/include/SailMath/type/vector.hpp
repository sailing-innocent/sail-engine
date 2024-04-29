#pragma once

/**
 * @file vector.hpp
 * @brief the vector class
 * @author sailing-innocent
 * @date 2024-04-29
 */

#include "SailMath/type/arithemetic.hpp"
#include <EASTL/array.h>

namespace sail {

template<ArithemeticT T, size_t N>
class Vector {
public:
	Vector() = default;

	// getter
	[[nodiscard]] T& operator[](size_t index) const noexcept {
		return m_data[index];
	}

private:
	eastl::array<T, N> m_data;
};

}// namespace sail