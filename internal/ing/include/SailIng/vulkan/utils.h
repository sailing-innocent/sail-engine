#pragma once
/**
 * @file: app/vulkan/vk_utils.h
 * @author: sailing-innocent
 * @create: 2022-10-23
 * @desp: the utility headers for vulkan applications
*/
#include "SailBase/config.h"

#define VK_USE_PLATFORM_WIN32_KHR
#include <vulkan/vulkan.h>
#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>
#define GLFW_EXPOSE_NATIVE_WIN32
#include <GLFW/glfw3native.h>
#define GLM_FORCE_RADIANS
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include "SailIng/util/file_loader.h"

#include <stdexcept>
#include <cstdlib>
#include <optional>
#include <set>
#include <cstdint>	// Necessary for uint32_t
#include <limits>	// Necessary for std::numeric_limits
#include <algorithm>// Necessary for std::clamp
#include <vector>
#include <iostream>
#include <array>
#include <chrono>

namespace sail::ing {

struct SAIL_ING_API VkOutVertex {
	glm::vec4 pos;
	glm::vec4 color;

	static VkVertexInputBindingDescription getBindingDescription() {
		VkVertexInputBindingDescription bindingDescription{};
		bindingDescription.binding = 0;
		bindingDescription.stride = sizeof(VkOutVertex);
		bindingDescription.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;// INSTANCE
		return bindingDescription;
	}

	static std::array<VkVertexInputAttributeDescription, 2> getAttributeDescriptions() {
		std::array<VkVertexInputAttributeDescription, 2> attributeDescriptions{};
		attributeDescriptions[0].binding = 0;
		attributeDescriptions[0].location = 0;
		attributeDescriptions[0].format = VK_FORMAT_R32G32_SFLOAT;
		// float: VK_FORMAT_R32_SFLOAT
		// vec2: VK_FORMAT_R32G32_SFLOAT
		// vec3: VK_FORMAT_R32G32B32_SFLOAT
		// vec4: VK_FORMAT_R32G32B32A32_SFLOAT
		// ivec2: VK_FORMAT_R32G32_SINT
		// uvec4: VK_FORMAT_R32G32B32A32_UINT
		// double: VK_FORMAT_R64_SFLOAT
		attributeDescriptions[0].offset = offsetof(VkOutVertex, pos);

		attributeDescriptions[1].binding = 0;
		attributeDescriptions[1].location = 1;
		attributeDescriptions[1].format = VK_FORMAT_R32G32B32_SFLOAT;
		attributeDescriptions[1].offset = offsetof(VkOutVertex, color);
		return attributeDescriptions;
	}
};

VkResult CreateDebugUtilsMessengerEXT(VkInstance instance,
									  const VkDebugUtilsMessengerCreateInfoEXT* pCreateInfo,
									  const VkAllocationCallbacks* pAllocator,
									  VkDebugUtilsMessengerEXT* pDebugMessenger);
void DestroyDebugUtilsMessengerEXT(VkInstance instance,
								   VkDebugUtilsMessengerEXT debugMessenger,
								   const VkAllocationCallbacks* pAllocator);
void populateDebugMessengerCreateInfo(VkDebugUtilsMessengerCreateInfoEXT& createInfo);

}// namespace sail::ing