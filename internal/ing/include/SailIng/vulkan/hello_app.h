#pragma once
/**
 * @file: include/app/vk_hello.hpp
 * @author: sailing-innocent
 * @create: 2022-10-23
*/

#include "basic_app.h"

namespace sail::ing {

class SAIL_ING_API INGVKHelloApp : public INGVKBasicApp {
public:
	INGVKHelloApp() = default;
	INGVKHelloApp(std::string _title, unsigned int _resw, unsigned int _resh)
		: INGVKBasicApp(_title, _resw, _resh) {
	}

protected:
	void cleanup() override;
	void createGraphicsPipeline() override;
	void createVertexBuffer() override;

protected:
	void recordCommandBuffer(VkCommandBuffer commandBuffer, uint32_t imageIndex) override;

protected:
	VkBuffer mVertexBuffer;
	VkDeviceMemory mVertexBufferMemory;

	const std::vector<VkOutVertex> mVertices = {
		{{0.0f, -0.5f, 0.0f, 1.0f}, {1.0f, 1.0f, 1.0f, 1.0f}},
		{{0.5f, 0.5f, 0.0f, 1.0f}, {0.0f, 1.0f, 0.0f, 1.0f}},
		{{-0.5f, 0.5f, 0.0f, 1.0f}, {0.0f, 0.0f, 1.0f, 1.0f}}};
	std::string mVertShaderPath{"assets/shaders/vulkan/basic.vert.spv"};
	std::string mFragShaderPath{"assets/shaders/vulkan/basic.frag.spv"};
};

}// namespace sail::ing