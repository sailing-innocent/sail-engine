#pragma once

/**
 * @file quaternion.hpp
 * @brief The Quaternion
 * @author sailing-innocent
 * @date 2024-04-29
 */

#include "SailMath/type/arithemetic.hpp"
#include <EASTL/array.h>
#include <cmath>

namespace sail {

template<RealT T>
class Quaternion {
public:
	bool is_normalized = false;
	Quaternion() = default;
	Quaternion(T x, T y, T z, T w) noexcept : m_x(x), m_y(y), m_z(z), m_w(w) {}
	Quaternion(const Quaternion<T>& q) noexcept {
		m_x = q.x;
		m_y = q.y;
		m_z = q.z;
		m_w = q.w;
	}
	Quaternion(Quaternion<T>&& q) noexcept {
		m_x = q.x;
		m_y = q.y;
		m_z = q.z;
		m_w = q.w;
	}
	~Quaternion() = default;

	// operators

	Quaternion<T>& operator=(const Quaternion<T>& q) noexcept {
		this = q;
	}
	Quaternion<T>& operator=(Quaternion<T>&& q) noexcept {
		this = eastl::move(q);
	}

	Quaternion<T> operator+(const Quaternion<T>& q) const noexcept {
		return Quaternion<T>(m_x + q.x, m_y + q.y, m_z + q.z, m_w + q.w);
	}
	Quaternion<T> operator-(const Quaternion<T>& q) const noexcept {
		return Quaternion<T>(m_x - q.x, m_y - q.y, m_z - q.z, m_w - q.w);
	}
	Quaternion<T> operator*(const Quaternion<T>& q) const noexcept {
		return Quaternion<T>(
			m_w * q.x + m_x * q.w + m_y * q.z - m_z * q.y,
			m_w * q.y - m_x * q.z + m_y * q.w + m_z * q.x,
			m_w * q.z + m_x * q.y - m_y * q.x + m_z * q.w,
			m_w * q.w - m_x * q.x - m_y * q.y - m_z * q.z);
	}
	Quaternion<T> operator*(T s) const noexcept {
		return Quaternion<T>(m_x * s, m_y * s, m_z * s, m_w * s);
	}
	Quaternion<T> operator/(T s) const noexcept {
		return Quaternion<T>(m_x / s, m_y / s, m_z / s, m_w / s);
	}

	bool operator==(const Quaternion<T>& q) const noexcept {
		return m_x == q.x && m_y == q.y && m_z == q.z && m_w == q.w;
	}
	bool operator!=(const Quaternion<T>& q) const noexcept {
		return m_x != q.x || m_y != q.y || m_z != q.z || m_w != q.w;
	}

	// functions
	bool normalize() noexcept {
		T len = m_x * m_x + m_y * m_y + m_z * m_z + m_w * m_w;
		if (len == static_cast<T>(0.0)) {
			return false;
		}
		len = static_cast<T>(1.0) / std::sqrt(len);
		m_x *= len;
		m_y *= len;
		m_z *= len;
		m_w *= len;
		is_normalized = true;
		return true;
	}

	Quaternion<T> conjugate() const noexcept {
		return Quaternion<T>(-m_x, -m_y, -m_z, m_w);
	}

	Quaternion<T> inverse() const noexcept {
		return conjugate() / (m_x * m_x + m_y * m_y + m_z * m_z + m_w * m_w);
	}

	Quaternion<T> rotate(const Quaternion<T>& q) const noexcept {
		return *this * q * inverse();
	}

	// setter
	[[nodiscard]] T& x() noexcept { return m_x; }
	[[nodiscard]] T& y() noexcept { return m_y; }
	[[nodiscard]] T& z() noexcept { return m_z; }
	[[nodiscard]] T& w() noexcept { return m_w; }

	// getter
	[[nodiscard]] T x() const noexcept { return m_x; }
	[[nodiscard]] T y() const noexcept { return m_y; }
	[[nodiscard]] T z() const noexcept { return m_z; }
	[[nodiscard]] T w() const noexcept { return m_w; }

private:
	T m_x = static_cast<T>(0.0);
	T m_y = static_cast<T>(0.0);
	T m_z = static_cast<T>(0.0);
	T m_w = static_cast<T>(1.0);
};

}// namespace sail
