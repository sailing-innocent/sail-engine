#pragma once
/**
 * @file: dummy_app/canvas
 * @author: sailing-innocent
 * @create: 2022-10-24
 * @desp: the canvas app for multiple static
 * @history:
 *  - 2022-10-24: create
 *  - 2022-11-13: try multiple pipelines
*/

#include "hello.h"

namespace sail::vk {

class SAIL_VK_API VKCanvasApp : public VKHelloApp {
public:
	VKCanvasApp() = default;
	VKCanvasApp(const std::string& _vertShaderPath, const std::string& _fragShaderPath);
	bool setVertex(std::vector<float> vfloat, size_t size);
	bool setIndex(std::vector<uint16_t> vu16, size_t size);

protected:
	void initVulkan() override;
	void cleanup() override;
	void createVertexBuffer() override;
	void createIndexBuffer() override;
	void createGraphicsPipeline() override;
	void drawFrame() override;
	// pipeline2
	void createGraphicsPipeline2();

protected:
	void recordCommandBuffer(VkCommandBuffer commandBuffer, uint32_t imageIndex) override;

	VkBuffer mIndexBuffer;
	VkDeviceMemory mIndexBufferMemory;
	std::vector<VkOutVertex> mVertices = {};
	std::vector<uint16_t> mIndices = {};

	std::string mVertShaderPath{"assets/shaders/vulkan/basic.vert.spv"};
	std::string mFragShaderPath{"assets/shaders/vulkan/basic.frag.spv"};
	// pipeline2
	VkPipelineLayout mPipelineLayout2;
	VkPipeline mGraphicsPipeline2;
};

// what is configurable for a pipeline
// vertShaderCode
// fragShaderCode
// --> shaderStages
// dynamicState
// VkOutVertex
// inputAssembly.topology
// viewport/scissor
// rasteriazer mode
// multisampling
// color BlendAttachment
// color Blending
// pipelineLayout <- for uniforms
// pipeline info <- mPipelineLayout mRenderPass

}// namespace sail::vk