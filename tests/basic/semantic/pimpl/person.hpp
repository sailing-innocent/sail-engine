#include <memory>

namespace sail::test {

class Person {
public:
	Person();
	~Person();
	[[nodiscard]] int id() const;

private:
	struct Impl;// hide details in impl
	std::unique_ptr<Impl> pimpl_;
};

}// namespace sail::test