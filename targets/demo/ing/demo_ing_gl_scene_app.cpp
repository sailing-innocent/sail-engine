#include "SailIng/opengl/scene_app.h"

using namespace sail;

int main() {
	ing::INGGLSceneApp app{};
	app.load_mesh("assets/models/bunny.obj");
	app.init();
	while (app.tick()) {
	}
	app.terminate();
	return 0;
}