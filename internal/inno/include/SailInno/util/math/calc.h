#pragma once

namespace sail::inno::math {
inline int imax(int a, int b) { return a > b ? a : b; }
inline bool is_power_of_two(int x) { return (x & (x - 1)) == 0; }
inline int floor_pow_2(int n) {
#ifdef WIN32
	return 1 << (int)logb((float)n);
#else
	int exp;
	frexp((float)n, &exp);
	return 1 << (exp - 1);
#endif
}
}// namespace sail::inno::math
