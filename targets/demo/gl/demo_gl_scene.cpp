#include "SailGL/dummy_app/scene.h"

using namespace sail;

int main() {
	gl::GLSceneDummyApp app{};
	app.load_mesh("assets/models/bunny.obj");
	app.init();
	while (app.tick()) {
	}
	app.terminate();
	return 0;
}