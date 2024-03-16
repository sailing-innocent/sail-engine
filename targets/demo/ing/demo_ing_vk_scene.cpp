
#include "SailIng/vulkan/scene_app.h"

using namespace sail;
int main() {
	const std::string vertShaderPath{"assets/shaders/vulkan/scene.vert.spv"};
	const std::string fragShaderPath{"assets/shaders/vulkan/scene.frag.spv"};
	ing::INGVKSceneApp app{vertShaderPath, fragShaderPath};
	// clang-format off
    std::vector<float> vf = {
        -0.5f, 0.0f, 0.0f, 1.0f, 1.0f, 0.0f, 0.0f, 1.0f,
        0.5f, 0.0f, 0.0f, 1.0f, 0.0f, 1.0f, 0.0f, 1.0f,
        0.0f, 0.5f, 0.0f, 1.0f, 0.0f, 0.0f, 1.0f, 1.0f,
        0.0f, -0.5f, 0.0f, 1.0f, 1.0f, 1.0f, 0.0f, 1.0f,
    };
	// clang-format on
	std::vector<uint16_t> iu = {0, 1, 2, 0, 3, 1};
	if (!app.setVertex(vf, vf.size())) {
		std::cerr << "input vertex failed" << std::endl;
	}
	if (!app.setIndex(iu, iu.size())) {
		std::cerr << "input index failed" << std::endl;
	};

	app.init();
	app.run();
	app.terminate();

	return 0;
}
