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
	while (app.tick()) {
	}
	return 0;
}
