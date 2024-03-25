#pragma once
/**
 * @file math/vector.h
 * @brief the vector math
 * @date 2023-11-12
 * @author sailing-innocent
*/

#include <array>

namespace sail {

template<typename T, int I>
class Vector {
public:
	//ctor
	Vector();
	Vector(const std::array<T, I>& data);
	Vector(const Vector<T, I>& other);
	Vector(Vector<T, I>&& other);
	Vector(const std::initializer_list<T> val);
	Vector<T, I>& operator=(const Vector<T, I>& rhs);
	Vector<T, I>& operator=(Vector<T, I>&& rhs);
	~Vector() {}

	// size
	int size() const { return I; }
	// get
	T operator[](int index) const;
	// set
	T& operator[](int index);

	// +=,-=,*=,/=,=
	Vector<T, I>& operator+=(const Vector<T, I>& rhs);
	Vector<T, I>& operator-=(const Vector<T, I>& rhs);
	Vector<T, I>& operator*=(const Vector<T, I>& rhs);
	Vector<T, I>& operator/=(const Vector<T, I>& rhs);
	// -
	friend Vector<T, I> operator-(const Vector<T, I>& rhs) {
		Vector<T, I> result;
		for (int i = 0; i < I; i++) {
			result[i] = -rhs[i];
		}
		return result;
	}

	// +-*/
	friend Vector<T, I> operator+(const Vector<T, I>& lhs, const Vector<T, I>& rhs) {
		Vector<T, I> result;
		for (int i = 0; i < I; i++) {
			result[i] = lhs[i] + rhs[i];
		}
		return result;
	}

	friend Vector<T, I> operator-(const Vector<T, I>& lhs, const Vector<T, I>& rhs) {
		Vector<T, I> result;
		for (int i = 0; i < I; i++) {
			result[i] = lhs[i] - rhs[i];
		}
		return result;
	}

	friend Vector<T, I> operator*(const Vector<T, I>& lhs, const Vector<T, I>& rhs) {
		Vector<T, I> result;
		for (int i = 0; i < I; i++) {
			result[i] = lhs[i] * rhs[i];
		}
		return result;
	}

	friend Vector<T, I> operator/(const Vector<T, I>& lhs, const Vector<T, I>& rhs) {
		Vector<T, I> result;
		for (int i = 0; i < I; i++) {
			result[i] = lhs[i] / rhs[i];
		}
		return result;
	}

	// equal
	friend bool operator==(const Vector<T, I>& lhs, const Vector<T, I>& rhs) {
		for (int i = 0; i < I; i++) {
			if (lhs[i] != rhs[i]) {
				return false;
			}
		}
		return true;
	}

	// T v[T] ops
	friend Vector<T, I> operator+(const Vector<T, I>& lhs, const T& rhs) {
		Vector<T, I> result;
		for (int i = 0; i < I; i++) {
			result[i] = lhs[i] + rhs;
		}
		return result;
	}

	friend Vector<T, I> operator+(const T& lhs, const Vector<T, I>& rhs) {
		Vector<T, I> result;
		for (int i = 0; i < I; i++) {
			result[i] = lhs + rhs[i];
		}
		return result;
	}

	friend Vector<T, I> operator-(const Vector<T, I>& lhs, const T& rhs) {
		Vector<T, I> result;
		for (int i = 0; i < I; i++) {
			result[i] = lhs[i] - rhs;
		}
		return result;
	}
	friend Vector<T, I> operator-(const T& lhs, const Vector<T, I>& rhs) {
		Vector<T, I> result;
		for (int i = 0; i < I; i++) {
			result[i] = lhs - rhs[i];
		}
		return result;
	}

	friend Vector<T, I> operator*(const Vector<T, I>& lhs, const T& rhs) {
		Vector<T, I> result;
		for (int i = 0; i < I; i++) {
			result[i] = lhs[i] * rhs;
		}
		return result;
	}
	friend Vector<T, I> operator*(const T& lhs, const Vector<T, I>& rhs) {
		Vector<T, I> result;
		for (int i = 0; i < I; i++) {
			result[i] = lhs * rhs[i];
		}
		return result;
	}
	friend Vector<T, I> operator/(const Vector<T, I>& lhs, const T& rhs) {
		Vector<T, I> result;
		for (int i = 0; i < I; i++) {
			result[i] = lhs[i] / rhs;
		}
		return result;
	}
	friend Vector<T, I> operator/(const T& lhs, const Vector<T, I>& rhs) {
		Vector<T, I> result;
		for (int i = 0; i < I; i++) {
			result[i] = lhs / rhs[i];
		}
		return result;
	}

	// T v[T] singleton ops
	Vector<T, I>& operator+=(const T val);
	Vector<T, I>& operator-=(const T val);
	Vector<T, I>& operator*=(const T val);
	Vector<T, I>& operator/=(const T val);

	// special ops
	// dot
	friend T dot(const Vector<T, I>& lhs, const Vector<T, I>& rhs) {
		T result = 0;
		for (int i = 0; i < I; i++) {
			result += lhs[i] * rhs[i];
		}
		return result;
	}
	// cross
	friend Vector<T, I> cross(const Vector<T, I>& lhs, const Vector<T, I>& rhs) {
		Vector<T, I> result;
		for (int i = 0; i < I; i++) {
			result[i] = lhs[(i + 1) % I] * rhs[(i + 2) % I] - lhs[(i + 2) % I] * rhs[(i + 1) % I];
		}
		return result;
	}
	// norm
	const double norm() const;
	// normalize

	friend Vector<double, 3> normalize(const Vector<T, I>& vec) {
		Vector<double, 3> result;
		double norm = vec.norm();
		for (int i = 0; i < I; i++) {
			result[i] = vec[i] / norm;
		}
		return result;
	}

private:
	std::array<T, I> m_data;
};

}// namespace sail

#include "vector.inl"
