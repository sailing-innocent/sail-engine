#include <glad/gl.h>
#include <GLFW/glfw3.h>

#include "SailIng/opengl/terrain/core/buffers.h"
#include "SailIng/opengl/terrain/core/texture.h"

namespace ing::terrain {

unsigned int create_frame_buffer() {
	unsigned int frame_buffer_handle;
	glGenFramebuffers(1, &frame_buffer_handle);
	glBindFramebuffer(GL_FRAMEBUFFER, frame_buffer_handle);
	glDrawBuffer(GL_COLOR_ATTACHMENT0);

	return frame_buffer_handle;
}

FrameBufferObject::FrameBufferObject(int W_, int H_)
	: W(W_), H(H_) {
}

void FrameBufferObject::bind() {}

}// namespace ing::terrain