/**
 * @file demo_gl_primitive.cpp
 * @brief The Demo Suite for GL Primitive App
 * @author sailing-innocent
 * @date 2024-05-01
 */

#include "SailGL/dummy_app/primitive.h"
#include "SailGL/util/gl_primitive.h"
#include <vector>
using namespace sail::gl;
using std::vector;
int main() {
	GLPoint p1(0.5f, -0.5f, 0.0f);
	GLPoint p2(-0.5f, -0.5f, 0.0f);
	GLPoint p3(0.0f, 0.5f, 0.0f);
	GLPoint p4(0.0f, -0.7f, 0.0f);
	GLPoint p5(0.0f, 0.6f, 0.0f);
	vector<float> blue = {0.0f, 0.0f, 1.0f, 1.0f};
	p4.setColor(blue);
	p1.setColor(blue);
	GLTriangle tr1(p1, p2, p3);
	GLTriangle tr2(p1, p4, p2);
	GLTriangle tr3(p1, p3, p5);
	vector<GLTriangle> _trlist = {tr1, tr2};
	GLTriangleList trlist{_trlist};
	GLPoint p6(-0.8f, 0.0f, 0.0f);
	GLPoint p7(0.8f, 0.0f, 0.0f);
	GLLine axis_x(p6, p7);
	GLPoint p8(0.0f, -0.8f, 0.0f);
	GLPoint p9(0.0f, 0.8f, 0.0f);
	GLLine axis_y(p8, p9);

	// app
	GLPrimitiveApp app;
	app.addTriangles(trlist);
	app.addTriangle(tr3);
	app.addLine(axis_x);
	app.addLine(axis_y);

	app.init();
	while (app.tick()) {
	}
	app.terminate();
	return 0;
}
