/**
 * @file app/cuda/basic_app.cpp
 * @author sailing-innocent
 * @date 2023-03-25
 * @brief the basic INGCUDAApp class
 */
#ifdef SAIL_ING_CUDA
#ifdef SAIL_ING_GL

#include "SailIng/cuda/basic_app.h"
#include "SailCu/demo/simple2d.h"

namespace sail::ing {

void INGCUDAApp::init_buffers() {
	// init shaders
	// init shaders
	std::string _vertPath = "assets/shaders/learnogl/basic.vert";
	std::string _fragPath = "assets/shaders/learnogl/basic.frag";
	GLShader newShader(_vertPath, _fragPath);
	m_shader = newShader;

	glGenVertexArrays(1, &m_VAO);
	glBindVertexArray(m_VAO);
	glGenBuffers(1, &m_pixels_VBO);
	glBindBuffer(GL_ARRAY_BUFFER, m_pixels_VBO);
	unsigned int size = m_resw * m_resh * 8 * sizeof(float);
	glBufferData(GL_ARRAY_BUFFER, size, 0, GL_DYNAMIC_DRAW);

	glVertexAttribPointer(0, 4, GL_FLOAT, GL_FALSE, 8 * sizeof(float), (void*)(0));
	glEnableVertexAttribArray(0);
	glVertexAttribPointer(1, 4, GL_FLOAT, GL_FALSE, 8 * sizeof(float), (void*)(4 * sizeof(float)));
	glEnableVertexAttribArray(1);

	glBindBuffer(GL_ARRAY_BUFFER, 0);

	cudaSetDevice(m_cuda_device);
	cudaGraphicsGLRegisterBuffer(
		&m_pixelsVBO_CUDA,
		m_pixels_VBO,
		cudaGraphicsMapFlagsWriteDiscard);
}

bool INGCUDAApp::tick(int count) {
	process_input_callback(m_window);

	cuda_shader();

	// Render from buffer object
	glClearColor(0.2f, 0.3f, 0.3f, 1.0f);
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	glBindVertexArray(m_VAO);
	m_shader.use();

	glBindBuffer(GL_ARRAY_BUFFER, m_pixels_VBO);
	glDrawArrays(GL_POINTS, 0, m_resw * m_resh);
	// glDrawArrays(GL_TRIANGLES, 0, 3);
	// glDrawArrays(GL_POINTS, 0, 3);
	glfwSwapBuffers(m_window);
	glfwPollEvents();

	return !glfwWindowShouldClose(m_window);
}

void INGCUDAApp::cuda_shader() {
	float timeValue = static_cast<float>(glfwGetTime());
	float* pixels;
	cudaGraphicsMapResources(1, &m_pixelsVBO_CUDA, 0);

	size_t num_bytes;
	cudaGraphicsResourceGetMappedPointer((void**)&pixels, &num_bytes, m_pixelsVBO_CUDA);

	// launch kernel
	sail::cu::Simple2DShader::sine_wave(pixels, timeValue, m_resh, m_resw);
	cudaGraphicsUnmapResources(1, &m_pixelsVBO_CUDA, 0);
}

void INGCUDAApp::destroy_buffers() {
	cudaGraphicsUnregisterResource(m_pixelsVBO_CUDA);
	glDeleteBuffers(1, &m_pixels_VBO);
}

}// namespace sail::ing

#endif// SAIL_ING_GL
#endif// SAIL_ING_CUDA