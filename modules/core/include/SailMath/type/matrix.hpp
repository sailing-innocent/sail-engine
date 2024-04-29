#pragma once
/**
 * @file matrix.hpp
 * @brief the matrix class
 * @author sailing-innocent
 * @date 2024-04-29
 */

#include "SailMath/type/arithemetic.hpp"

#include <EASTL/array.h>
#include <cmath>

namespace sail {

template<ArithemeticT T, size_t M, size_t N>
class Matrix {
public:
	Matrix() = default;
	explicit Matrix(T value) {
		for (size_t i = 0; i < M; ++i) {
			for (size_t j = 0; j < N; ++j) {
				m_data[i][j] = value;
			}
		}
	}
	explicit Matrix(const eastl::array<eastl::array<T, N>, M>& data) : m_data(data) {}
	// param list
	template<typename... Args>
	explicit Matrix(Args... args) : m_data{args...} {
		static_assert(sizeof...(args) == M, "the size of args should be equal to M");
	}
	// copy
	Matrix(const Matrix& other) = default;
	Matrix& operator=(const Matrix& other) = default;
	// move
	Matrix(Matrix&& other) noexcept = default;
	Matrix& operator=(Matrix&& other) noexcept = default;

	[[nodiscard]] eastl::array<T, N>& operator[](size_t index) noexcept {
		return m_data[index];
	}
	[[nodiscard]] const eastl::array<T, N>& operator[](size_t index) const noexcept {
		return m_data[index];
	}

	// operator
	friend Matrix operator+(const Matrix& lhs, const Matrix& rhs) {
		Matrix result;
		for (size_t i = 0; i < M; ++i) {
			for (size_t j = 0; j < N; ++j) {
				result[i][j] = lhs[i][j] + rhs[i][j];
			}
		}
		return result;
	}
	friend Matrix operator-(const Matrix& lhs, const Matrix& rhs) {
		Matrix result;
		for (size_t i = 0; i < M; ++i) {
			for (size_t j = 0; j < N; ++j) {
				result[i][j] = lhs[i][j] - rhs[i][j];
			}
		}
		return result;
	}
	// scalar multiply
	friend Matrix operator*(const Matrix& lhs, T rhs) {
		Matrix result;
		for (size_t i = 0; i < M; ++i) {
			for (size_t j = 0; j < N; ++j) {
				result[i][j] = lhs[i][j] * rhs;
			}
		}
		return result;
	}

	// scalar divide
	friend Matrix operator/(const Matrix& lhs, T rhs) {
		auto rhs_inv = static_cast<T>(1) / rhs;
		return lhs * rhs_inv;
	}

	// matrix multiply
	// 3-layer nested loop
	template<size_t P>
	friend Matrix<T, M, P> operator*(const Matrix& lhs, const Matrix<T, N, P>& rhs) {
		Matrix<T, M, P> result;
		for (size_t i = 0; i < M; ++i) {
			for (size_t j = 0; j < P; ++j) {
				for (size_t k = 0; k < N; ++k) {
					result[i][j] += lhs[i][k] * rhs[k][j];
				}
			}
		}
		return result;
	}

private:
	eastl::array<eastl::array<T, N>, M> m_data;
};

}// namespace sail