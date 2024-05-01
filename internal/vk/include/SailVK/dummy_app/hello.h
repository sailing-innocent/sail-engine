#pragma once
/**
 * @file hello.h
 * @brief Hello App
 * @author sailing-innocent
 * @date 2022-10-26
 */
#include "basic.h"

namespace sail::vk {

class SAIL_VK_API VKHelloApp : public VKBasicApp {
public:
	VKHelloApp() = default;
	VKHelloApp(std::string _title, unsigned int _resw, unsigned int _resh)
		: VKBasicApp(_title, _resw, _resh) {
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

}// namespace sail::vk