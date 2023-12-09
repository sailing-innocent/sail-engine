#include "SailIng/vulkan/hello_app.h"
using namespace sail;

int main() {
	ing::INGVKHelloApp app{"INGVKHelloApp", 1600, 900};
	app.init();
	while (app.tick(0)) {
	}
	app.terminate();
	return 0;
}