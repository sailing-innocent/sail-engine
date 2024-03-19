#include "person.hpp"
#include <string>
#include <memory>

namespace sail::test {

struct Person::Impl {
	std::string name = "hello";
	int id = 1;
};

Person::Person() : pimpl_(std::make_unique<Impl>()) {}
Person::~Person() = default;

int Person::id() const {
	return pimpl_->id;
}

}// namespace sail::test