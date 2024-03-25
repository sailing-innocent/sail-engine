/**
 * @file math/vector.inl
 * @brief the vector math implementation
 * @date 2023-11-12
 * @author sailing-innocent
*/

namespace sail {

template<typename T, int I>
Vector<T, I>::Vector() {
	m_data = std::array<T, I>();
	// fill 0
	for (int i = 0; i < I; i++) {
		m_data[i] = 0;
	}
}

template<typename T, int I>
Vector<T, I>::Vector(const std::initializer_list<T> val_list) {
	m_data = std::array<T, I>();
	// fill val
	auto it_begin = val_list.begin();
	if (val_list.size() != I) {
		if (val_list.size() == 1) {
			for (int i = 0; i < I; i++) {
				m_data[i] = *it_begin;
			}
		}
	} else {
		for (auto it = it_begin; it != val_list.end(); it++) {
			m_data[it - it_begin] = *it;
		}
	}
}
template<typename T, int I>
Vector<T, I>::Vector(const std::array<T, I>& data) {
	// deep clone
	m_data = std::array<T, I>();
	for (int i = 0; i < I; i++) {
		m_data[i] = data[i];
	}
}

template<typename T, int I>
Vector<T, I>::Vector(const Vector<T, I>& other) {
	// deep clone
	m_data = std::array<T, I>();
	for (int i = 0; i < I; i++) {
		m_data[i] = other[i];
	}
}
// move
template<typename T, int I>
Vector<T, I>::Vector(Vector<T, I>&& other) {
	// shallow clone
	m_data = other.m_data;
}

// get
template<typename T, int I>
T Vector<T, I>::operator[](int index) const {
	return m_data[index];
};

// set
template<typename T, int I>
T& Vector<T, I>::operator[](int index) {
	return m_data[index];
};

// +=
template<typename T, int I>
Vector<T, I>& Vector<T, I>::operator+=(const Vector<T, I>& rhs) {
	for (int i = 0; i < I; i++) {
		m_data[i] += rhs[i];
	}
	return *this;
}

// -=
template<typename T, int I>
Vector<T, I>& Vector<T, I>::operator-=(const Vector<T, I>& rhs) {
	for (int i = 0; i < I; i++) {
		m_data[i] -= rhs[i];
	}
	return *this;
}

// *=
template<typename T, int I>
Vector<T, I>& Vector<T, I>::operator*=(const Vector<T, I>& rhs) {
	for (int i = 0; i < I; i++) {
		m_data[i] *= rhs[i];
	}
	return *this;
}

// /=
template<typename T, int I>
Vector<T, I>& Vector<T, I>::operator/=(const Vector<T, I>& rhs) {
	for (int i = 0; i < I; i++) {
		m_data[i] /= rhs[i];
	}
	return *this;
}

}// namespace sail

// T v[T] operators
namespace sail {

template<typename T, int I>
Vector<T, I>& Vector<T, I>::operator+=(const T val) {
	for (int i = 0; i < I; i++) {
		m_data[i] += val;
	}
	return *this;
}

template<typename T, int I>
Vector<T, I>& Vector<T, I>::operator-=(const T val) {
	for (int i = 0; i < I; i++) {
		m_data[i] -= val;
	}
	return *this;
}

template<typename T, int I>
Vector<T, I>& Vector<T, I>::operator*=(const T val) {
	for (int i = 0; i < I; i++) {
		m_data[i] *= val;
	}
	return *this;
}

template<typename T, int I>
Vector<T, I>& Vector<T, I>::operator/=(const T val) {
	for (int i = 0; i < I; i++) {
		m_data[i] /= val;
	}
	return *this;
}

template<typename T, int I>
Vector<T, I>& Vector<T, I>::operator=(const Vector<T, I>& rhs) {
	// deep clone
	for (int i = 0; i < I; i++) {
		m_data[i] = rhs[i];
	}
	return *this;
}

template<typename T, int I>
Vector<T, I>& Vector<T, I>::operator=(Vector<T, I>&& rhs) {
	// shallow clone
	m_data = rhs.m_data;
	return *this;
}

}// namespace sail

// special ops
namespace sail {

// dot (friend impl)
// cross (friend impl)
// norm
template<typename T, int I>
const double Vector<T, I>::norm() const {
	double result = 0.0;
	for (int i = 0; i < I; i++) {
		result += static_cast<double>(m_data[i]) * m_data[i];
	}
	return sqrt(result);
}

// normalize

}// namespace sail