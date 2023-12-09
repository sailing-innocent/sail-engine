#pragma once
/**
 * @file: include/app/vk_scene.hpp
 * @author: sailing-innocent
 * @create: 2022-10-24
 * @desp: the scene app for 3D view
*/

#include "canvas_app.h"
#include "SailIng/util/timer.h"

namespace sail::ing {

const int MAX_FRAMES_IN_FLIGHT = 2;

struct UniformBufferObject {
	glm::mat4 model;
	glm::mat4 view;
	glm::mat4 proj;
};

class SAIL_ING_API INGVKSceneApp : public INGVKCanvasApp {
public:
	INGVKSceneApp() = default;
	INGVKSceneApp(const std::string& _vert_p, const std::string& _frag_p)
		: INGVKCanvasApp(_vert_p, _frag_p) {
	}

protected:
	void cleanup();
	void mainLoop();
	void createDescriptorSetLayout();
	void createGraphicsPipeline();
	void createUniformBuffers();
	void createDescriptorPool();
	void createDescriptorSets();

	void drawFrame(sail::Timer<double>& tmr);

protected:
	void recordCommandBuffer(VkCommandBuffer commandBuffer, uint32_t imageIndex);
	void updateUniformBuffer(uint32_t currentImage, sail::Timer<double>& tmr);
	void calculateFrameStats();

protected:
	VkDescriptorSetLayout mDescriptorSetLayout;
	VkPipelineLayout mPipelineLayout;

	std::vector<VkBuffer> mUniformBuffers;
	std::vector<VkDeviceMemory> mUniformBuffersMemory;
	std::vector<void*> mUniformBuffersMapped;

	VkDescriptorPool mDescriptorPool;
	std::vector<VkDescriptorSet> mDescriptorSets;
	uint32_t mCurrentFrame = 0;

	sail::Timer<double> mTimer;
	bool mAppPaused = false;
	const float mFPS = 60;
};// class INGVKSceneApp

}// namespace sail::ing