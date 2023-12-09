#include "test_util.h"

#include <vector>
#include <iostream>

namespace sail::test {

class Person {
	int _age;

public:
	Person(int age) : _age(age) { std::cout << "Constructing a Person"
											<< std::endl; }
	Person(const Person& p) : _age(p._age) { std::cout << "Copy Constructing a Person"
													   << std::endl; }
	Person(Person&& p) : _age(p._age) { std::cout << "Move Constructing a Person"
												  << std::endl; }
};

}// namespace sail::test

TEST_SUITE("basic::containers") {
	TEST_CASE("vector_push_back") {
		std::vector<int> v = {1, 2};
		REQUIRE(v.size() == 2);
		v.push_back(3);
		REQUIRE(v.size() == 3);
	}
	TEST_CASE("vector_emplace_back") {
		std::vector<int> v = {1, 2};
		REQUIRE(v.size() == 2);
		v.emplace_back(3);
		REQUIRE(v.size() == 3);
		// v.emplace_back(4,5);
		// REQUIRE(v.size() == 5);
	}
	TEST_CASE("vector_diff_push_back_emplace_back") {
		std::vector<sail::test::Person> v;
		v.reserve(5);
		auto p = sail::test::Person(10);
		v.push_back(p);
		v.emplace_back(20);
		v.emplace_back(std::move(p));
		// v.push_back(std::move(p));
	}
}