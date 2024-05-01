#include "SailVK/dummy_app/basic.h"
using namespace sail::vk;

int main(int argc, char* argv[]) {
	VKBasicApp app{"VKBasicApp", 1600, 900};
	app.init();
	while (app.tick(0)) {
	}
	app.terminate();
	return 0;
}