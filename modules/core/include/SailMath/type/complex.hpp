#pragma once
/**
 * @file complex.hpp
 * @brief Complex Number for Math
 * @author sailing-innocent
 * @date 2024-04-29
 */

#include "SailMath/type/arithemetic.hpp"
namespace sail {

template<RealT T>
class Complex {
public:
	Complex() = default;
	explicit Complex(T real) : m_real(real), m_imag(static_cast<T>(0)) {}
	explicit Complex(T real, T imag) : m_real(real), m_imag(imag) {}

	// getter
	[[nodiscard]] T real() const noexcept { return m_real; }
	[[nodiscard]] T imag() const noexcept { return m_imag; }
	// setter
	T& real() noexcept { return m_real; }
	T& imag() noexcept { return m_imag; }

	// operator
	T norm() const noexcept {
		return m_real * m_real + m_imag * m_imag;
	}
	friend Complex operator+(const Complex& lhs, const Complex& rhs) {
		return Complex(lhs.m_real + rhs.m_real, lhs.m_imag + rhs.m_imag);
	}
	friend Complex operator-(const Complex& lhs, const Complex& rhs) {
		return Complex(lhs.m_real - rhs.m_real, lhs.m_imag - rhs.m_imag);
	}
	friend Complex operator*(const Complex& lhs, const Complex& rhs) {
		return Complex(lhs.m_real * rhs.m_real - lhs.m_imag * rhs.m_imag,
					   lhs.m_real * rhs.m_imag + lhs.m_imag * rhs.m_real);
	}
	friend Complex operator/(const Complex& lhs, const Complex& rhs) {
		T denominator = rhs.norm();
		return Complex((lhs.m_real * rhs.m_real + lhs.m_imag * rhs.m_imag) / denominator,
					   (lhs.m_imag * rhs.m_real - lhs.m_real * rhs.m_imag) / denominator);
	}

private:
	T m_real;
	T m_imag;
};

using ComplexF = Complex<float>;
using ComplexD = Complex<double>;

}// namespace sail