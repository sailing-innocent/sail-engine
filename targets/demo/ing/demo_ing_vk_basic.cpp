#include "SailIng/vulkan/basic_app.h"
using namespace sail;

int main() {
	ing::INGVKBasicApp app{"INGVKBasicApp", 1600, 900};
	app.init();
	while (app.tick(0)) {
	}
	app.terminate();
	return 0;
}