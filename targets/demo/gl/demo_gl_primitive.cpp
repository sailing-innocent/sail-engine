/**
 * @file demo_gl_primitive.cpp
 * @brief The Demo Suite for GL Primitive App
 * @author sailing-innocent
 * @date 2024-05-01
 */

#include "SailGL/dummy_app/primitive.h"

int main() {
	sail::gl::GLPrimitiveApp app;
	app.init();
	while (app.tick()) {
	}
	app.terminate();
	return 0;
}
