#include "SailVK/dummy_app/canvas.h"

using namespace sail::vk;

namespace sail::test {

bool genIndex(std::vector<uint16_t>& vu) {
	const size_t size = 6;
	vu.resize(size);
	uint16_t indi[size] = {
		0, 1, 2, 2, 3, 0};
	for (auto i = 0; i < size; i++) {
		vu[i] = indi[i];
	}
	return true;
}

bool genVertex(std::vector<float>& vf) {
	vf = {
		-0.5f, -0.5f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 1.0f,
		0.5f, -0.5f, 0.0f, 1.0f, 0.0f, 1.0f, 0.0f, 1.0f,
		0.5f, 0.5f, 0.0f, 1.0f, 0.0f, 0.0f, 1.0f, 1.0f,
		-0.5f, 0.5f, 0.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f};
	return true;
}

}// namespace sail::test

using namespace sail;
int main() {
	VKCanvasApp app{};
	std::vector<float> vertices;
	test::genVertex(vertices);

	std::vector<uint16_t> indices;
	test::genIndex(indices);

	if (!app.setVertex(vertices, vertices.size())) {
		std::cout << "init verttices failed" << std::endl;
	}
	if (!app.setIndex(indices, indices.size())) {
		std::cout << "init indices failed" << std::endl;
	}

	app.init();
	while (app.tick(0)) {
	}
	app.terminate();
	return 0;
}
