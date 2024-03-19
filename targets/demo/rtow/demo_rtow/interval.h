#pragma once

namespace sail::rtow {

class interval {
public:
	double m_min;
	double m_max;
	interval() : m_min(+infinity), m_max(-infinity) {}
	interval(double _min, double _max) : m_min(_min), m_max(_max) {}

	bool contains(double x) const {
		return m_min <= x && x <= m_max;
	}
	bool surrounds(double x) const {
		return m_min < x && x < m_max;
	}
	double clamp(double x) const {
		if (x < m_min) {
			return m_min;
		}
		if (x > m_max) {
			return m_max;
		}
		return x;
	}

	static const interval empty, universe;
};

const static interval empty{+infinity, -infinity};
const static interval universe{-infinity, +infinity};
}// namespace sail::rtow