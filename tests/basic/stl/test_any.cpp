#include "test_util.h"
#include <any>
#include <typeindex>
#include <vector>

namespace sail::test {

class Dog {
public:
	[[nodiscard]] int run() noexcept {
		return 1;
	}
};

class Bird {
public:
	[[nodiscard]] int fly() noexcept {
		return 2;
	}
};

class Fish {
public:
	[[nodiscard]] int swim() noexcept {
		return 3;
	}
};

[[nodiscard]] int out_any(std::any const& val) noexcept {
	static auto dog_type = std::type_index(typeid(Dog));
	static auto bird_type = std::type_index(typeid(Bird));
	static auto fish_type = std::type_index(typeid(Fish));
	const auto val_type = std::type_index(val.type());
	if (val_type == dog_type) {
		return std::any_cast<Dog>(val).run();
	}
	if (val_type == bird_type) {
		return std::any_cast<Bird>(val).fly();
	}
	if (val_type == fish_type) {
		return std::any_cast<Fish>(val).swim();
	}
	return -1;
}

int test_any() {
	Dog dog;
	Bird bird;
	Fish fish;
	std::vector<std::any> animals = {dog, bird, fish};
	std::vector<int> results = {1, 2, 3};

	for (size_t i = 0; i < animals.size(); ++i) {
		CHECK(out_any(animals[i]) == results[i]);
	}
	return 0;
}

}// namespace sail::test

TEST_SUITE("basic::stl") {
	TEST_CASE("any") {
		REQUIRE(sail::test::test_any() == 0);
	}
}