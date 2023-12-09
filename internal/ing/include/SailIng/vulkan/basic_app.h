#pragma once
/**
 * @file app/vulkan/vk_basic_app.h
 * @author sailing-innocent
 * @date 2023-02-25
 * @brief the baic Vulkan App
 */
#include "SailBase/config.h"

#include "SailIng/opengl/pure_app.h"
#include "utils.h"

namespace sail::ing {

struct SAIL_ING_API QueueFamilyIndices {
	std::optional<uint32_t> graphicsFamily;
	std::optional<uint32_t> presentFamily;

	bool isComplete() {
		return graphicsFamily.has_value() && presentFamily.has_value();
	}
};

struct SwapChainSupportDetails {
	VkSurfaceCapabilitiesKHR capabilities;
	std::vector<VkSurfaceFormatKHR> formats;
	std::vector<VkPresentModeKHR> presentModes;
};

class SAIL_ING_API INGVKBasicApp : public INGGLPureApp {
public:
	INGVKBasicApp() = default;
	INGVKBasicApp(std::string _title,
				  unsigned int _resw,
				  unsigned int _resh) : INGGLPureApp(_title, _resw, _resh) {
	}
	virtual ~INGVKBasicApp() {
		destroy_buffers();
		destroy_window();
	}

public:
	virtual void init() override;
	virtual bool tick(int count) override;
	virtual void terminate() override;
	virtual void wait();
	virtual void run();

public:
	// accessors
	GLFWwindow* get_window() { return m_window; }
	unsigned int get_resw() { return m_resw; }
	unsigned int get_resh() { return m_resh; }

protected:
	// procedure
	virtual void initWindow();
	virtual void initVulkan();
	virtual void mainLoop();
	virtual void cleanup();
	virtual void createInstance();
	virtual void pickPhysicalDevice();
	virtual void createLogicalDevice();
	virtual void createSurface();
	virtual void createSwapChain();
	virtual void createImageViews();
	virtual void createRenderPass();
	virtual void createDescriptorSetLayout();
	virtual void createGraphicsPipeline();
	virtual void createFramebuffers();
	virtual void createCommandPool();
	virtual void createTextureImage();
	virtual void createVertexBuffer();
	virtual void createIndexBuffer();
	virtual void createUniformBuffers();
	virtual void createDescriptorPool();
	virtual void createDescriptorSets();
	virtual void createCommandBuffer();
	virtual void createSyncObjects();
	virtual void drawFrame();
	virtual void setupDebugMessenger();

protected:
	// utils methods
	virtual bool checkValidationLayerSupport();
	virtual std::vector<const char*> getRequiredExtensions();
	static VKAPI_ATTR VkBool32 VKAPI_CALL debugCallback(
		VkDebugUtilsMessageSeverityFlagBitsEXT messageSeverity,
		VkDebugUtilsMessageTypeFlagsEXT messageType,
		const VkDebugUtilsMessengerCallbackDataEXT* pCallbackData,
		void* pUserData) {
		if (messageSeverity >= VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT) {
			std::cerr << "validation layer: " << pCallbackData->pMessage << std::endl;
		}
		return VK_FALSE;
	}
	virtual QueueFamilyIndices findQueueFamilies(VkPhysicalDevice device);
	virtual bool isDeviceSuitable(VkPhysicalDevice device);
	virtual bool checkDeviceExtensionSupport(VkPhysicalDevice device);
	virtual SwapChainSupportDetails querySwapChainSupport(VkPhysicalDevice device);
	virtual VkSurfaceFormatKHR chooseSwapSurfaceFormat(const std::vector<VkSurfaceFormatKHR>& availableFormats);
	virtual VkPresentModeKHR chooseSwapPresentMode(const std::vector<VkPresentModeKHR>& availablePresentModes);
	virtual VkExtent2D chooseSwapExtent(const VkSurfaceCapabilitiesKHR& capabilities);
	virtual VkShaderModule createShaderModule(const std::vector<char>& code);
	virtual void recordCommandBuffer(VkCommandBuffer commandBuffer, uint32_t imageIndex);
	virtual void createBuffer(VkDeviceSize size, VkBufferUsageFlags usage, VkMemoryPropertyFlags properties, VkBuffer& buffer, VkDeviceMemory& bufferMemory);
	virtual uint32_t findMemoryType(uint32_t typeFilter, VkMemoryPropertyFlags properties);
	virtual void copyBuffer(VkBuffer srcBuffer, VkBuffer dstBuffer, VkDeviceSize size0);

protected:
	VkInstance mInstance;
	VkDebugUtilsMessengerEXT mDebugMessenger;

	const std::vector<const char*> mValidationLayers = {
		"VK_LAYER_KHRONOS_validation"};

	VkPhysicalDevice mPhysicalDevice = VK_NULL_HANDLE;// Physical Device Handle
	VkDevice mDevice;								  // Logical Device
	VkQueue mGraphicsQueue;							  // Directly fetch graphics queue
	VkSurfaceKHR mSurface;							  // On windows it is called VK_KHR_win32_surface; glfwGetRequiredInstanceExtensions
	VkQueue mPresentQueue;							  // Present Queue

	const std::vector<const char*> mDeviceExtensions = {
		VK_KHR_SWAPCHAIN_EXTENSION_NAME};

	VkSwapchainKHR mSwapChain;
	std::vector<VkImage> mSwapChainImages;
	VkFormat mSwapChainImageFormat;
	VkExtent2D mSwapChainExtent;
	std::vector<VkImageView> mSwapChainImageViews;

	VkRenderPass mRenderPass;
	VkPipelineLayout mPipelineLayout;
	VkPipeline mGraphicsPipeline;

	std::vector<VkFramebuffer> mSwapChainFramebuffers;
	VkCommandPool mCommandPool;
	VkCommandBuffer mCommandBuffer;

#ifdef NDEBUG
	bool mEnableValidationLayers = false;
#else
	bool mEnableValidationLayers = true;
#endif

	VkSemaphore mImageAvailableSemaphore;
	VkSemaphore mRenderFinishedSemaphore;
	VkFence mInFlightFence;
};// class INGVKBasicApp

}// namespace sail::ing