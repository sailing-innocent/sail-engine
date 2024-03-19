#include "test_util.h"
#include <vector>
#include <variant>

namespace sail::test {

template<typename Derived>
class Animal {
public:
	[[nodiscard]] int move() noexcept {
		return static_cast<Derived*>(this)->move_impl();
	}

private:
	Derived const& derived() const {
		return static_cast<Derived const&>(*this);
	}
};

class Dog : public Animal<Dog> {
public:
	int move_impl() {
		return 1;
	}
};

class Bird : public Animal<Bird> {
public:
	int move_impl() {
		return 2;
	}
};

class Fish : public Animal<Fish> {
public:
	int move_impl() {
		return 3;
	}
};

int test_crtp() {
	Dog dog;
	Bird bird;
	Fish fish;
	CHECK(dog.move() == 1);
	CHECK(bird.move() == 2);
	CHECK(fish.move() == 3);

	using pAnimal_t = std::variant<Dog*, Bird*, Fish*>;
	std::vector<pAnimal_t> animals{&dog, &bird, &fish};
	std::vector<int> results{1, 2, 3};

	for (size_t i = 0; i < animals.size(); ++i) {
		CHECK(std::visit([](auto& animal) { return animal->move(); }, animals[i]) == results[i]);
	}
	return 0;
}

}// namespace sail::test

TEST_SUITE("basic::design") {
	TEST_CASE("crtp") {
		REQUIRE(sail::test::test_crtp() == 0);
	}
}