#include "test_util.h"
#include <coroutine>
#include <exception>
#include <stdexcept>
#include <cstdint>
#include <iostream>

namespace sail::test {

template<typename T>
struct Generator {
	// compiler recognize coroutine by co_yield keyword
	struct promise_type;
	using handle_type = std::coroutine_handle<promise_type>;

	struct promise_type {
		T value_;
		std::exception_ptr exception_;
		Generator get_return_object() {
			return Generator(handle_type::from_promise(*this));
		}
		std::suspend_always initial_suspend() { return {}; }
		std::suspend_always final_suspend() noexcept { return {}; }
		void unhandled_exception() {
			exception_ = std::current_exception();
		}
		template<std::convertible_to<T> From>
		std::suspend_always yield_value(From&& value) {
			value_ = std::forward<From>(value);
			return {};
		}
		void return_void() {}
	};

	handle_type h_;

	Generator(handle_type h) : h_(h) {}
	~Generator() { h_.destroy(); }

	explicit operator bool() {
		fill();
		return !h_.done();
	}
	T operator()() {
		fill();
		full_ = false;
		return std::move(h_.promise().value_);
	}

private:
	bool full_ = false;
	void fill() {
		if (!full_) {
			h_();
			if (h_.promise().exception_) {
				std::rethrow_exception(h_.promise().exception_);
			}
			full_ = true;
		}
	}
};

Generator<std::uint64_t>
fibonacci(unsigned int n) {
	if (n == 0) {
		co_return;
	}
	if (n > 94) {
		throw std::runtime_error("n is too large");
	}

	co_yield 0;

	if (n == 1) {
		co_return;
	}

	co_yield 1;

	if (n == 2) {
		co_return;
	}

	std::uint64_t a = 0;
	std::uint64_t b = 1;

	for (unsigned int i = 2; i < n; ++i) {
		std::uint64_t c = a + b;
		co_yield c;
		a = b;
		b = c;
	}
}

int test_coroutine() {
	try {
		auto gen = fibonacci(10);
		int a = 0;
		int b = 1;
		int c;
		for (int i = 0; i < 10; i++) {
			if (i == 0) {
				CHECK(gen() == 0);
			} else if (i == 1) {
				CHECK(gen() == 1);
			} else {
				c = a + b;
				CHECK(gen() == c);
				a = b;
				b = c;
			}
		}
	} catch (const std::exception& e) {
		std::cerr << "Exception: " << e.what() << std::endl;
		return 1;
	} catch (...) {
		std::cerr << "Unknown exception" << std::endl;
		return 1;
	}
	return 0;
}

}// namespace sail::test

TEST_SUITE("basic::stl") {
	TEST_CASE("coroutine") {
		REQUIRE(sail::test::test_coroutine() == 0);
	}
}