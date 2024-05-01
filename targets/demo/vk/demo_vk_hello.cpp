/**
 * @file demo_vk_hello.cpp
 * @brief The Demo of Vulkan Hello App
 * @author sailing-innocent
 * @date 2024-05-01
 */

#include "SailVK/dummy_app/hello.h"

using namespace sail::vk;

int main() {
	VKHelloApp app{"VKHelloApp", 1600, 900};
	app.init();
	while (app.tick(0)) {
	}
	app.terminate();
	return 0;
}