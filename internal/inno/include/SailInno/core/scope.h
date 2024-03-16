#pragma once
#include <luisa/dsl/sugar.h>

namespace sail::inno::dsl {
class scope {
public:
	static void escape() { $break; }
	static void require(luisa::compute::Bool escape_if_false) {
		$if(!escape_if_false) {
			$break;
		};
	};
};

template<typename F>
inline void operator%(scope, F&& f) {
	$loop {
		f();
		$break;
	};
}

// scope for escaping at any point
#define $scope ::sphere::dsl::scope{} % [&]
#define $require(b) ::sphere::dsl::scope::require((b))
#define $escape ::sphere::dsl::scope::escape()

}// namespace sail::inno::dsl