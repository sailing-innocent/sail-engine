#pragma once
/** 
 * @file terrain/buffers.h
 * @author sailing-innocent
 * @date 2023-08-30
 * @brief the terrain buffers
 */
#include "SailBase/config.h"

namespace ing::terrain {

unsigned int SAIL_ING_API create_frame_buffer();

class SAIL_ING_API FrameBufferObject {
public:
	FrameBufferObject(int w, int h);
	unsigned int FBO, render_buffer, depth_tex;
	unsigned int tex;
	unsigned int get_color_attachment_tex(int i);
	void bind();

private:
	int W, H;
	int n_color_attachments;
	unsigned int* m_color_attachments;
};

}// namespace ing::terrain