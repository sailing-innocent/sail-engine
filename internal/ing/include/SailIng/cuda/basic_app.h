#pragma once
/**
 * @file app/cuda/basic_app.h
 * @author sailing-innocent
 * @date 2023-03-25
 * @brief the basic INGCUDAApp class
 */
#ifdef SAIL_ING_CUDA
#ifdef SAIL_ING_GL

#include "SailIng/opengl/pure_app.h"
#include "SailIng/opengl/gl_shader.h"
#include <cuda.h>

#include <cuda_runtime_api.h>
#include <cuda_gl_interop.h>

namespace sail::ing {

class SAIL_ING_API INGCUDAApp : public INGGLPureApp {
public:
	INGCUDAApp() = default;
	INGCUDAApp(std::string _title,
			   unsigned int _resw,
			   unsigned int _resh) : INGGLPureApp(_title, _resw, _resh) {
	}
	virtual ~INGCUDAApp() {}
	bool tick(int count = 0) override;

protected:
	virtual void cuda_shader();
	void init_buffers() override;
	void destroy_buffers() override;

	// resource
	unsigned int m_cuda_device = 0;
	unsigned int m_pixels_VBO;
	unsigned int m_VAO;
	cudaGraphicsResource_t m_pixelsVBO_CUDA;
	GLShader m_shader;
};

}// namespace sail::ing

#endif
#endif