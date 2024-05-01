/**
 * @file demo_gl_pure.cpp
 * @brief The Demo for Pure OpenGL App
 * @author sailing-innocent
 * @date 2024-05-01
 */

#include "SailGL/dummy_app/pure.h"

int main() {
	sail::gl::GLPureDummyApp app{};
	app.init();
	int max_iter = 20;
	while (max_iter-- > 0) {
		app.tick();
	}

	return 0;
}
