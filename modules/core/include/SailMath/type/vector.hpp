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
	explicit Vector(T value) {
		for (size_t i = 0; i < N; ++i) {
			m_data[i] = value;
		}
	}
	explicit Vector(const eastl::array<T, N>& data) : m_data(data) {}
	// param list
	template<typename... Args>
	explicit Vector(Args... args) : m_data{args...} {
		static_assert(sizeof...(args) == N, "the size of args should be equal to N");
	}
	// copy
	Vector(const Vector& other) = default;
	Vector& operator=(const Vector& other) = default;
	// move
	Vector(Vector&& other) noexcept = default;
	Vector& operator=(Vector&& other) noexcept = default;

	[[nodiscard]] T& operator[](size_t index) noexcept {
		return m_data[index];
	}
	[[nodiscard]] const T& operator[](size_t index) const noexcept {
		return m_data[index];
	}

	// operator
	friend Vector operator+(const Vector& lhs, const Vector& rhs) {
		Vector result;
		for (size_t i = 0; i < N; ++i) {
			result[i] = lhs[i] + rhs[i];
		}
		return result;
	}
	friend Vector operator-(const Vector& lhs, const Vector& rhs) {
		Vector result;
		for (size_t i = 0; i < N; ++i) {
			result[i] = lhs[i] - rhs[i];
		}
		return result;
	}
	// scalar multiply
	friend Vector operator*(const Vector& lhs, T rhs) {
		Vector result;
		for (size_t i = 0; i < N; ++i) {
			result[i] = lhs[i] * rhs;
		}
		return result;
	}
	friend Vector operator*(T lhs, const Vector& rhs) {
		Vector result;
		for (size_t i = 0; i < N; ++i) {
			result[i] = lhs * rhs[i];
		}
		return result;
	}
	friend Vector operator/(const Vector& lhs, T rhs) {
		Vector result;
		for (size_t i = 0; i < N; ++i) {
			result[i] = lhs[i] / rhs;
		}
		return result;
	}
	// dot product
	friend T dot(const Vector& lhs, const Vector& rhs) {
		T result = 0;
		for (size_t i = 0; i < N; ++i) {
			result += lhs[i] * rhs[i];
		}
		return result;
	}
	// cross product
	friend Vector cross(const Vector& lhs, const Vector& rhs) {
		static_assert(N == 3, "cross product is only defined for 3D vector");
		return Vector(lhs[1] * rhs[2] - lhs[2] * rhs[1],
					  lhs[2] * rhs[0] - lhs[0] * rhs[2],
					  lhs[0] * rhs[1] - lhs[1] * rhs[0]);
	}

private:
	eastl::array<T, N> m_data;
};

}// namespace sail