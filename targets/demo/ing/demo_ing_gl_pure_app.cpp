#include "SailIng/opengl/pure_app.h"

int main() {
	bool background = false;
	sail::ing::INGGLPureApp app{};
	app.init();
	while (app.tick()) {
	}
	return 0;
}