#include "SailIng/vulkan/canvas_app.h"

namespace sail::ing {

void INGVKCanvasApp::initVulkan() {
	createInstance();

	setupDebugMessenger();

	createSurface();
	pickPhysicalDevice();
	createLogicalDevice();
	createSwapChain();
	createImageViews();
	createRenderPass();

	createDescriptorSetLayout();

	createGraphicsPipeline();
	createGraphicsPipeline2();

	createFramebuffers();
	createCommandPool();

	createTextureImage();

	createVertexBuffer();
	createIndexBuffer();
	createUniformBuffers();
	createDescriptorPool();
	createDescriptorSets();

	createCommandBuffer();
	createSyncObjects();
}

INGVKCanvasApp::INGVKCanvasApp(const std::string& _vertShaderPath, const std::string& _fragShaderPath) : mVertShaderPath(_vertShaderPath),
																										 mFragShaderPath(_fragShaderPath) {}

bool INGVKCanvasApp::setVertex(std::vector<float> vfloat, size_t size) {
	if (size % 8 != 0) { return false; }
	mVertices.resize(size);
	for (auto i = 0; i < size / 8; i++) {
		mVertices[i] = {
			{vfloat[8 * i + 0], vfloat[8 * i + 1], vfloat[8 * i + 2], vfloat[8 * i + 3]},
			{vfloat[8 * i + 4], vfloat[8 * i + 5], vfloat[8 * i + 6], vfloat[8 * i + 7]}};
	}
	return true;
}

bool INGVKCanvasApp::setIndex(std::vector<uint16_t> vu16, size_t size) {
	mIndices.resize(size);
	for (auto i = 0; i < size; i++) {
		mIndices[i] = vu16[i];
	}
	return true;
}

void INGVKCanvasApp::createIndexBuffer() {
	VkDeviceSize bufferSize = sizeof(mIndices[0]) * mIndices.size();

	// STAGING BUFFERS ON CPU
	VkBuffer stagingBuffer;
	VkDeviceMemory stagingBufferMemory;
	createBuffer(bufferSize,
				 VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
				 VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
					 VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
				 stagingBuffer,
				 stagingBufferMemory);
	void* data;
	vkMapMemory(mDevice, stagingBufferMemory, 0, bufferSize, 0, &data);
	memcpy(data, mIndices.data(), (size_t)bufferSize);
	vkUnmapMemory(mDevice, stagingBufferMemory);

	// CREATE GPU BUFFER
	createBuffer(bufferSize,
				 VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_INDEX_BUFFER_BIT,
				 VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
				 mIndexBuffer,
				 mIndexBufferMemory);
	copyBuffer(stagingBuffer, mIndexBuffer, bufferSize);

	vkDestroyBuffer(mDevice, stagingBuffer, nullptr);
	vkFreeMemory(mDevice, stagingBufferMemory, nullptr);
}

void INGVKCanvasApp::drawFrame() {
	vkWaitForFences(mDevice, 1, &mInFlightFence, VK_TRUE, UINT64_MAX);
	vkResetFences(mDevice, 1, &mInFlightFence);
	// acquire an image from swapchain
	uint32_t imageIndex;
	vkAcquireNextImageKHR(mDevice, mSwapChain, UINT64_MAX, mImageAvailableSemaphore, VK_NULL_HANDLE, &imageIndex);
	vkResetCommandBuffer(mCommandBuffer, 0);
	recordCommandBuffer(mCommandBuffer, imageIndex);
	// we can now submit it
	VkSubmitInfo submitInfo{};
	submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
	VkSemaphore waitSemaphores[] = {mImageAvailableSemaphore};
	VkPipelineStageFlags waitStages[] = {VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT};
	submitInfo.waitSemaphoreCount = 1;
	submitInfo.pWaitSemaphores = waitSemaphores;
	submitInfo.pWaitDstStageMask = waitStages;
	submitInfo.commandBufferCount = 1;
	submitInfo.pCommandBuffers = &mCommandBuffer;

	VkSemaphore signalSemaphores[] = {mRenderFinishedSemaphore};
	submitInfo.signalSemaphoreCount = 1;
	submitInfo.pSignalSemaphores = signalSemaphores;

	// the last parameter is optional, let us know when is safe
	if (vkQueueSubmit(mGraphicsQueue, 1, &submitInfo, mInFlightFence) != VK_SUCCESS) {
		std::runtime_error("failed to submit draw command buffer!");
	}

	// presentation
	VkPresentInfoKHR presentInfo{};
	presentInfo.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
	presentInfo.waitSemaphoreCount = 1;
	presentInfo.pWaitSemaphores = signalSemaphores;

	VkSwapchainKHR swapChains[] = {mSwapChain};
	presentInfo.swapchainCount = 1;
	presentInfo.pSwapchains = swapChains;
	presentInfo.pImageIndices = &imageIndex;

	presentInfo.pResults = nullptr;
	vkQueuePresentKHR(mPresentQueue, &presentInfo);
}

void INGVKCanvasApp::recordCommandBuffer(VkCommandBuffer commandBuffer, uint32_t imageIndex) {
	VkCommandBufferBeginInfo beginInfo{};
	beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
	beginInfo.flags = 0;
	beginInfo.pInheritanceInfo = nullptr;// Optional

	if (vkBeginCommandBuffer(commandBuffer, &beginInfo) != VK_SUCCESS) {
		throw std::runtime_error("failed to begin recording command buffer!");
	}

	// create a render pass to bind a swap chain
	VkRenderPassBeginInfo renderPassInfo{};
	renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
	renderPassInfo.renderPass = mRenderPass;
	renderPassInfo.framebuffer = mSwapChainFramebuffers[imageIndex];
	renderPassInfo.renderArea.offset = {0, 0};
	renderPassInfo.renderArea.extent = mSwapChainExtent;

	VkClearValue clearColor = {{{0.0f, 0.0f, 0.0f, 1.0f}}};
	renderPassInfo.clearValueCount = 1;
	renderPassInfo.pClearValues = &clearColor;

	// begin render pass for each pipeline
	vkCmdBeginRenderPass(commandBuffer, &renderPassInfo, VK_SUBPASS_CONTENTS_INLINE);

	// bind the first pipeline, with its vertex and indices

	// ADD BUFFER
	VkBuffer vertexBuffers[] = {mVertexBuffer};
	VkDeviceSize offsets[] = {0};

	VkViewport viewport{};
	viewport.x = 0.0f;
	viewport.y = 0.0f;
	viewport.width = static_cast<float>(mSwapChainExtent.width);
	viewport.height = static_cast<float>(mSwapChainExtent.height);
	viewport.minDepth = 0.0f;
	viewport.maxDepth = 1.0f;
	vkCmdSetViewport(commandBuffer, 0, 1, &viewport);

	VkRect2D scissor{};
	scissor.offset = {0, 0};
	scissor.extent = mSwapChainExtent;
	vkCmdSetScissor(commandBuffer, 0, 1, &scissor);

	vkCmdBindVertexBuffers(commandBuffer, 0, 1, vertexBuffers, offsets);
	vkCmdBindIndexBuffer(commandBuffer, mIndexBuffer, 0, VK_INDEX_TYPE_UINT16);

	// bind triangle pipeline
	vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, mGraphicsPipeline);
	vkCmdDrawIndexed(commandBuffer, static_cast<uint32_t>(mIndices.size()), 1, 0, 0, 0);

	// bind line pipeline
	vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, mGraphicsPipeline2);
	vkCmdDrawIndexed(commandBuffer, static_cast<uint32_t>(mIndices.size()), 1, 0, 0, 0);

	vkCmdEndRenderPass(commandBuffer);
	if (vkEndCommandBuffer(commandBuffer) != VK_SUCCESS) {
		throw std::runtime_error("failed to record command buffer!");
	}
}

void INGVKCanvasApp::createVertexBuffer() {
	VkDeviceSize bufferSize = sizeof(mVertices[0]) * mVertices.size();
	VkBuffer stagingBuffer;
	VkDeviceMemory stagingBufferMemory;
	createBuffer(bufferSize,
				 VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
				 VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
				 stagingBuffer,
				 stagingBufferMemory);
	// copy the data
	void* data;
	vkMapMemory(mDevice, stagingBufferMemory, 0, bufferSize, 0, &data);
	memcpy(data, mVertices.data(), (size_t)bufferSize);
	vkUnmapMemory(mDevice, stagingBufferMemory);

	createBuffer(bufferSize,
				 VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
				 VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
				 mVertexBuffer,
				 mVertexBufferMemory);

	copyBuffer(stagingBuffer, mVertexBuffer, bufferSize);

	vkDestroyBuffer(mDevice, stagingBuffer, nullptr);
	vkFreeMemory(mDevice, stagingBufferMemory, nullptr);
}

void INGVKCanvasApp::cleanup() {
	vkDestroySemaphore(mDevice, mImageAvailableSemaphore, nullptr);
	vkDestroySemaphore(mDevice, mRenderFinishedSemaphore, nullptr);
	vkDestroyFence(mDevice, mInFlightFence, nullptr);
	vkDestroyCommandPool(mDevice, mCommandPool, nullptr);
	for (auto framebuffer : mSwapChainFramebuffers) {
		vkDestroyFramebuffer(mDevice, framebuffer, nullptr);
	}
	vkDestroyPipeline(mDevice, mGraphicsPipeline, nullptr);
	vkDestroyPipelineLayout(mDevice, mPipelineLayout, nullptr);

	vkDestroyPipeline(mDevice, mGraphicsPipeline2, nullptr);
	vkDestroyPipelineLayout(mDevice, mPipelineLayout2, nullptr);

	vkDestroyRenderPass(mDevice, mRenderPass, nullptr);
	for (auto imageView : mSwapChainImageViews) {
		vkDestroyImageView(mDevice, imageView, nullptr);
	}
	vkDestroySwapchainKHR(mDevice, mSwapChain, nullptr);

	// DESTROY BUFFER
	vkDestroyBuffer(mDevice, mVertexBuffer, nullptr);
	vkFreeMemory(mDevice, mVertexBufferMemory, nullptr);
	vkDestroyBuffer(mDevice, mIndexBuffer, nullptr);
	vkFreeMemory(mDevice, mIndexBufferMemory, nullptr);

	vkDestroyDevice(mDevice, nullptr);
	if (mEnableValidationLayers) {
		DestroyDebugUtilsMessengerEXT(mInstance, mDebugMessenger, nullptr);
	}
	vkDestroySurfaceKHR(mInstance, mSurface, nullptr);
	vkDestroyInstance(mInstance, nullptr);
	glfwDestroyWindow(m_window);
	glfwTerminate();
}

void INGVKCanvasApp::createGraphicsPipeline() {
	// std::cout << "Happy Pipeline!" << std::endl;
	auto vertShaderCode = readfile(mVertShaderPath.c_str());
	auto fragShaderCode = readfile(mFragShaderPath.c_str());

	VkShaderModule vertShaderModule = createShaderModule(vertShaderCode);
	VkShaderModule fragShaderModule = createShaderModule(fragShaderCode);

	VkPipelineShaderStageCreateInfo vertShaderStageInfo{};
	vertShaderStageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
	vertShaderStageInfo.stage = VK_SHADER_STAGE_VERTEX_BIT;
	vertShaderStageInfo.module = vertShaderModule;
	vertShaderStageInfo.pName = "main";

	VkPipelineShaderStageCreateInfo fragShaderStageInfo{};
	fragShaderStageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
	fragShaderStageInfo.stage = VK_SHADER_STAGE_FRAGMENT_BIT;
	fragShaderStageInfo.module = fragShaderModule;
	fragShaderStageInfo.pName = "main";

	VkPipelineShaderStageCreateInfo shaderStages[] = {vertShaderStageInfo, fragShaderStageInfo};

	std::vector<VkDynamicState> dynamicStates = {
		VK_DYNAMIC_STATE_VIEWPORT,
		VK_DYNAMIC_STATE_SCISSOR};

	VkPipelineDynamicStateCreateInfo dynamicState{};
	dynamicState.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
	dynamicState.dynamicStateCount = static_cast<uint32_t>(dynamicStates.size());
	dynamicState.pDynamicStates = dynamicStates.data();

	VkPipelineVertexInputStateCreateInfo vertexInputInfo{};
	vertexInputInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
	auto bindingDescription = VkOutVertex::getBindingDescription();
	auto attributeDescriptions = VkOutVertex::getAttributeDescriptions();
	vertexInputInfo.vertexBindingDescriptionCount = 1;
	vertexInputInfo.vertexAttributeDescriptionCount = static_cast<uint32_t>(attributeDescriptions.size());
	vertexInputInfo.pVertexBindingDescriptions = &bindingDescription;
	vertexInputInfo.pVertexAttributeDescriptions = attributeDescriptions.data();

	// Input Assembly
	VkPipelineInputAssemblyStateCreateInfo inputAssembly{};
	inputAssembly.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
	inputAssembly.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
	inputAssembly.primitiveRestartEnable = VK_FALSE;

	// Viewport
	VkViewport viewport{};
	viewport.x = 0.0f;
	viewport.y = 0.0f;
	viewport.height = (float)mSwapChainExtent.height;
	viewport.width = (float)mSwapChainExtent.width;
	viewport.minDepth = 0.0f;
	viewport.maxDepth = 1.0f;

	// Scissor Rectangle
	VkRect2D scissor{};
	scissor.offset = {0, 0};
	scissor.extent = mSwapChainExtent;

	// view port state
	VkPipelineViewportStateCreateInfo viewportState{};
	viewportState.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
	viewportState.viewportCount = 1;
	viewportState.pViewports = &viewport;
	viewportState.scissorCount = 1;
	viewportState.pScissors = &scissor;

	// Rasterizer

	VkPipelineRasterizationStateCreateInfo rasterizer{};
	rasterizer.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
	rasterizer.depthClampEnable = VK_FALSE;
	rasterizer.rasterizerDiscardEnable = VK_FALSE;
	rasterizer.polygonMode = VK_POLYGON_MODE_FILL;// VK_POLYGON_MODE_LINE VK_POLYGON_MODE_POINT
	rasterizer.lineWidth = 1.0f;
	rasterizer.cullMode = VK_CULL_MODE_BACK_BIT;
	rasterizer.frontFace = VK_FRONT_FACE_CLOCKWISE;

	rasterizer.depthBiasEnable = VK_FALSE;
	rasterizer.depthBiasConstantFactor = 0.0f;
	rasterizer.depthBiasClamp = 0.0f;
	rasterizer.depthBiasSlopeFactor = 0.0f;

	// MultiSampling
	VkPipelineMultisampleStateCreateInfo multisampling{};
	multisampling.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
	multisampling.sampleShadingEnable = VK_FALSE;
	multisampling.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;
	multisampling.minSampleShading = 1.0f;		   // Optional
	multisampling.pSampleMask = nullptr;		   // Optional
	multisampling.alphaToCoverageEnable = VK_FALSE;// Optional
	multisampling.alphaToOneEnable = VK_FALSE;	   // Optional

	// Depth & Stencil Buffer

	// Color Belending
	VkPipelineColorBlendAttachmentState colorBlendAttachment{};
	colorBlendAttachment.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
	colorBlendAttachment.srcColorBlendFactor = VK_BLEND_FACTOR_ONE; // Optional
	colorBlendAttachment.dstColorBlendFactor = VK_BLEND_FACTOR_ZERO;// Optional
	colorBlendAttachment.colorBlendOp = VK_BLEND_OP_ADD;			// Optional
	colorBlendAttachment.srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE; // Optional
	colorBlendAttachment.dstAlphaBlendFactor = VK_BLEND_FACTOR_ZERO;// Optional
	colorBlendAttachment.alphaBlendOp = VK_BLEND_OP_ADD;			// Optional

	/*
    colorBlendAttachment.blendEnable = VK_TRUE;
    colorBlendAttachment.srcColorBlendFactor = VK_BLEND_FACTOR_SRC_ALPHA;
    colorBlendAttachment.dstColorBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
    colorBlendAttachment.colorBlendOp = VK_BLEND_OP_ADD;
    colorBlendAttachment.srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE;
    colorBlendAttachment.dstAlphaBlendFactor = VK_BLEND_FACTOR_ZERO;
    colorBlendAttachment.alphaBlendOp = VK_BLEND_OP_ADD;
    */

	VkPipelineColorBlendStateCreateInfo colorBlending{};
	colorBlending.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
	colorBlending.logicOpEnable = VK_FALSE;
	colorBlending.logicOp = VK_LOGIC_OP_COPY;// Optional
	colorBlending.attachmentCount = 1;
	colorBlending.pAttachments = &colorBlendAttachment;
	colorBlending.blendConstants[0] = 0.0f;// Optional
	colorBlending.blendConstants[1] = 0.0f;// Optional
	colorBlending.blendConstants[2] = 0.0f;// Optional
	colorBlending.blendConstants[3] = 0.0f;// Optional

	VkPipelineLayoutCreateInfo pipelineLayoutInfo{};
	pipelineLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
	pipelineLayoutInfo.setLayoutCount = 0;			 // Optional
	pipelineLayoutInfo.pSetLayouts = nullptr;		 // Optional
	pipelineLayoutInfo.pushConstantRangeCount = 0;	 // Optional
	pipelineLayoutInfo.pPushConstantRanges = nullptr;// Optional

	if (vkCreatePipelineLayout(mDevice, &pipelineLayoutInfo, nullptr, &mPipelineLayout) != VK_SUCCESS) {
		throw std::runtime_error("failed to create pipeline layout!");
	}

	// Finally, the pipeline
	VkGraphicsPipelineCreateInfo pipelineInfo{};
	pipelineInfo.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
	pipelineInfo.stageCount = 2;
	pipelineInfo.pStages = shaderStages;
	pipelineInfo.pVertexInputState = &vertexInputInfo;
	pipelineInfo.pInputAssemblyState = &inputAssembly;
	pipelineInfo.pViewportState = &viewportState;
	pipelineInfo.pRasterizationState = &rasterizer;
	pipelineInfo.pMultisampleState = &multisampling;
	pipelineInfo.pColorBlendState = &colorBlending;
	pipelineInfo.pDynamicState = &dynamicState;
	pipelineInfo.layout = mPipelineLayout;
	pipelineInfo.renderPass = mRenderPass;
	pipelineInfo.subpass = 0;
	pipelineInfo.basePipelineHandle = VK_NULL_HANDLE;
	pipelineInfo.basePipelineIndex = -1;

	if (vkCreateGraphicsPipelines(mDevice, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &mGraphicsPipeline) != VK_SUCCESS) {
		throw std::runtime_error("failed to create graphics pipeline!");
	}

	vkDestroyShaderModule(mDevice, fragShaderModule, nullptr);
	vkDestroyShaderModule(mDevice, vertShaderModule, nullptr);
}

void INGVKCanvasApp::createGraphicsPipeline2() {
	// std::cout << "Happy Pipeline!" << std::endl;
	auto vertShaderCode = readfile(mVertShaderPath.c_str());
	auto fragShaderCode = readfile(mFragShaderPath.c_str());

	VkShaderModule vertShaderModule = createShaderModule(vertShaderCode);
	VkShaderModule fragShaderModule = createShaderModule(fragShaderCode);

	VkPipelineShaderStageCreateInfo vertShaderStageInfo{};
	vertShaderStageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
	vertShaderStageInfo.stage = VK_SHADER_STAGE_VERTEX_BIT;
	vertShaderStageInfo.module = vertShaderModule;
	vertShaderStageInfo.pName = "main";

	VkPipelineShaderStageCreateInfo fragShaderStageInfo{};
	fragShaderStageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
	fragShaderStageInfo.stage = VK_SHADER_STAGE_FRAGMENT_BIT;
	fragShaderStageInfo.module = fragShaderModule;
	fragShaderStageInfo.pName = "main";

	VkPipelineShaderStageCreateInfo shaderStages[] = {vertShaderStageInfo, fragShaderStageInfo};

	std::vector<VkDynamicState> dynamicStates = {
		VK_DYNAMIC_STATE_VIEWPORT,
		VK_DYNAMIC_STATE_SCISSOR};

	VkPipelineDynamicStateCreateInfo dynamicState{};
	dynamicState.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
	dynamicState.dynamicStateCount = static_cast<uint32_t>(dynamicStates.size());
	dynamicState.pDynamicStates = dynamicStates.data();

	VkPipelineVertexInputStateCreateInfo vertexInputInfo{};
	vertexInputInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
	auto bindingDescription = VkOutVertex::getBindingDescription();
	auto attributeDescriptions = VkOutVertex::getAttributeDescriptions();
	vertexInputInfo.vertexBindingDescriptionCount = 1;
	vertexInputInfo.vertexAttributeDescriptionCount = static_cast<uint32_t>(attributeDescriptions.size());
	vertexInputInfo.pVertexBindingDescriptions = &bindingDescription;
	vertexInputInfo.pVertexAttributeDescriptions = attributeDescriptions.data();

	// Input Assembly
	VkPipelineInputAssemblyStateCreateInfo inputAssembly{};
	inputAssembly.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
	inputAssembly.topology = VK_PRIMITIVE_TOPOLOGY_LINE_LIST;// VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
	inputAssembly.primitiveRestartEnable = VK_FALSE;

	// Viewport
	VkViewport viewport{};
	viewport.x = 0.0f;
	viewport.y = 0.0f;
	viewport.height = (float)mSwapChainExtent.height;
	viewport.width = (float)mSwapChainExtent.width;
	viewport.minDepth = 0.0f;
	viewport.maxDepth = 1.0f;

	// Scissor Rectangle
	VkRect2D scissor{};
	scissor.offset = {0, 0};
	scissor.extent = mSwapChainExtent;

	// view port state
	VkPipelineViewportStateCreateInfo viewportState{};
	viewportState.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
	viewportState.viewportCount = 1;
	viewportState.pViewports = &viewport;
	viewportState.scissorCount = 1;
	viewportState.pScissors = &scissor;

	// Rasterizer

	VkPipelineRasterizationStateCreateInfo rasterizer{};
	rasterizer.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
	rasterizer.depthClampEnable = VK_FALSE;
	rasterizer.rasterizerDiscardEnable = VK_FALSE;
	rasterizer.polygonMode = VK_POLYGON_MODE_FILL;// VK_POLYGON_MODE_LINE VK_POLYGON_MODE_POINT
	rasterizer.lineWidth = 1.0f;
	rasterizer.cullMode = VK_CULL_MODE_BACK_BIT;
	rasterizer.frontFace = VK_FRONT_FACE_CLOCKWISE;

	rasterizer.depthBiasEnable = VK_FALSE;
	rasterizer.depthBiasConstantFactor = 0.0f;
	rasterizer.depthBiasClamp = 0.0f;
	rasterizer.depthBiasSlopeFactor = 0.0f;

	// MultiSampling
	VkPipelineMultisampleStateCreateInfo multisampling{};
	multisampling.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
	multisampling.sampleShadingEnable = VK_FALSE;
	multisampling.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;
	multisampling.minSampleShading = 1.0f;		   // Optional
	multisampling.pSampleMask = nullptr;		   // Optional
	multisampling.alphaToCoverageEnable = VK_FALSE;// Optional
	multisampling.alphaToOneEnable = VK_FALSE;	   // Optional

	// Depth & Stencil Buffer

	// Color Belending
	VkPipelineColorBlendAttachmentState colorBlendAttachment{};
	colorBlendAttachment.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
	colorBlendAttachment.srcColorBlendFactor = VK_BLEND_FACTOR_ONE; // Optional
	colorBlendAttachment.dstColorBlendFactor = VK_BLEND_FACTOR_ZERO;// Optional
	colorBlendAttachment.colorBlendOp = VK_BLEND_OP_ADD;			// Optional
	colorBlendAttachment.srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE; // Optional
	colorBlendAttachment.dstAlphaBlendFactor = VK_BLEND_FACTOR_ZERO;// Optional
	colorBlendAttachment.alphaBlendOp = VK_BLEND_OP_ADD;			// Optional

	/*
    colorBlendAttachment.blendEnable = VK_TRUE;
    colorBlendAttachment.srcColorBlendFactor = VK_BLEND_FACTOR_SRC_ALPHA;
    colorBlendAttachment.dstColorBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
    colorBlendAttachment.colorBlendOp = VK_BLEND_OP_ADD;
    colorBlendAttachment.srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE;
    colorBlendAttachment.dstAlphaBlendFactor = VK_BLEND_FACTOR_ZERO;
    colorBlendAttachment.alphaBlendOp = VK_BLEND_OP_ADD;
    */

	VkPipelineColorBlendStateCreateInfo colorBlending{};
	colorBlending.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
	colorBlending.logicOpEnable = VK_FALSE;
	colorBlending.logicOp = VK_LOGIC_OP_COPY;// Optional
	colorBlending.attachmentCount = 1;
	colorBlending.pAttachments = &colorBlendAttachment;
	colorBlending.blendConstants[0] = 0.0f;// Optional
	colorBlending.blendConstants[1] = 0.0f;// Optional
	colorBlending.blendConstants[2] = 0.0f;// Optional
	colorBlending.blendConstants[3] = 0.0f;// Optional

	VkPipelineLayoutCreateInfo pipelineLayoutInfo{};
	pipelineLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
	pipelineLayoutInfo.setLayoutCount = 0;			 // Optional
	pipelineLayoutInfo.pSetLayouts = nullptr;		 // Optional
	pipelineLayoutInfo.pushConstantRangeCount = 0;	 // Optional
	pipelineLayoutInfo.pPushConstantRanges = nullptr;// Optional

	if (vkCreatePipelineLayout(mDevice, &pipelineLayoutInfo, nullptr, &mPipelineLayout2) != VK_SUCCESS) {
		throw std::runtime_error("failed to create pipeline layout 2!");
	}

	// Finally, the pipeline
	VkGraphicsPipelineCreateInfo pipelineInfo{};
	pipelineInfo.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
	pipelineInfo.stageCount = 2;
	pipelineInfo.pStages = shaderStages;
	pipelineInfo.pVertexInputState = &vertexInputInfo;
	pipelineInfo.pInputAssemblyState = &inputAssembly;
	pipelineInfo.pViewportState = &viewportState;
	pipelineInfo.pRasterizationState = &rasterizer;
	pipelineInfo.pMultisampleState = &multisampling;
	pipelineInfo.pColorBlendState = &colorBlending;
	pipelineInfo.pDynamicState = &dynamicState;
	pipelineInfo.layout = mPipelineLayout2;
	pipelineInfo.renderPass = mRenderPass;
	pipelineInfo.subpass = 0;
	pipelineInfo.basePipelineHandle = VK_NULL_HANDLE;
	pipelineInfo.basePipelineIndex = -1;

	if (vkCreateGraphicsPipelines(mDevice, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &mGraphicsPipeline2) != VK_SUCCESS) {
		throw std::runtime_error("failed to create graphics pipeline 2!");
	}

	vkDestroyShaderModule(mDevice, fragShaderModule, nullptr);
	vkDestroyShaderModule(mDevice, vertShaderModule, nullptr);
}

}// namespace sail::ing
